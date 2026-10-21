"""Live facial emotion detection with Flask, combining an EfficientNet-B0 and a YOLOv8 emotion model."""
import json
import os
import queue
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, session
from torchvision import models

BASE_DIR = Path(__file__).resolve().parent
EFFNET_PATH = BASE_DIR / "models" / "emotion_efficientnet_b0.pth"
YOLO_PATH = BASE_DIR / "models" / "best.pt"
DB_PATH = BASE_DIR / "data" / "emotions.db"
MEDIA_DIR = BASE_DIR / "media"
FACE_DIR = MEDIA_DIR / "new_face"
SAVE_INTERVAL = float(os.environ.get("SAVE_INTERVAL", 1.0))  # seconds between saved faces, per visitor
EFFNET_WEIGHT = float(os.environ.get("EFFNET_WEIGHT", 0.5))  # YOLO gets the rest of the vote
DETECT_WIDTH = 320  # face detection runs on a smaller copy of the frame
TERMS_VERSION = "1.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


def load_secret_key():
    """Keep the session key on disk so visitors stay signed in across restarts."""
    path = BASE_DIR / "data" / "secret_key"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32))
    return path.read_text().strip()


app.secret_key = os.environ.get("SECRET_KEY") or load_secret_key()


# ---------- models ----------

def load_effnet():
    checkpoint = torch.load(EFFNET_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["class_names"]
    size = int(checkpoint.get("image_size", 224))
    mean = np.array(checkpoint.get("imagenet_mean", (0.485, 0.456, 0.406)), dtype=np.float32)
    std = np.array(checkpoint.get("imagenet_std", (0.229, 0.224, 0.225)), dtype=np.float32)

    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(classes))
    net.load_state_dict(checkpoint["model_state_dict"])
    return net.to(DEVICE).eval(), classes, size, mean, std


def load_yolo():
    from ultralytics import YOLO

    yolo = YOLO(str(YOLO_PATH))
    size = int(yolo.ckpt.get("train_args", {}).get("imgsz", 96))
    classes = [yolo.names[i] for i in sorted(yolo.names)]
    return yolo.model.fuse(verbose=False).float().to(DEVICE).eval(), classes, size


effnet, class_names, effnet_size, effnet_mean, effnet_std = load_effnet()
yolo, yolo_classes, yolo_size = load_yolo()
if yolo_classes != class_names:
    raise RuntimeError(f"The two models use different classes: {class_names} vs {yolo_classes}")

# The first forward pass is much slower than the rest, so do it now instead of on the first request.
with torch.inference_mode():
    effnet(torch.zeros(1, 3, effnet_size, effnet_size, device=DEVICE))
    yolo(torch.zeros(1, 3, yolo_size, yolo_size, device=DEVICE))

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(f"EfficientNet-B0 and YOLO loaded on {DEVICE}")


# ---------- storage ----------

def connect():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS visitors (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT NOT NULL,
                terms_version     TEXT NOT NULL,
                terms_accepted_at TEXT NOT NULL,
                user_agent        TEXT
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id         INTEGER NOT NULL REFERENCES visitors(id),
                name               TEXT NOT NULL,
                image_path         TEXT NOT NULL,
                emotion            TEXT NOT NULL,
                confidence         REAL NOT NULL,
                scores             TEXT NOT NULL,
                effnet_emotion     TEXT NOT NULL,
                effnet_confidence  REAL NOT NULL,
                yolo_emotion       TEXT NOT NULL,
                yolo_confidence    REAL NOT NULL,
                box                TEXT NOT NULL,
                created_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS predictions_visitor ON predictions(visitor_id);
        """)


def safe_filename_part(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40] or "user"


def save_worker():
    """Write faces to disk and the database off the request thread, so saving never slows predictions."""
    db = connect()
    while True:
        visitor_id, name, face_rgb, face, now = save_queue.get()
        try:
            filename = f"{safe_filename_part(name)}_{now:%Y-%m-%d_%H-%M-%S-%f}"[:-3] + ".jpg"
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise ValueError("could not encode the face image")
            (FACE_DIR / filename).write_bytes(jpeg.tobytes())
            db.execute(
                """INSERT INTO predictions (visitor_id, name, image_path, emotion, confidence, scores,
                       effnet_emotion, effnet_confidence, yolo_emotion, yolo_confidence, box, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (visitor_id, name, f"media/new_face/{filename}", face["emotion"], face["confidence"],
                 json.dumps(face["scores"]), face["effnet"]["emotion"], face["effnet"]["confidence"],
                 face["yolo"]["emotion"], face["yolo"]["confidence"], json.dumps(face["box"]),
                 now.isoformat(timespec="milliseconds")),
            )
            db.commit()
        except Exception as err:
            db.rollback()
            print(f"Could not save face: {err}")


init_db()
save_queue = queue.Queue(maxsize=50)
threading.Thread(target=save_worker, daemon=True).start()
last_save = {}  # visitor id -> time of their last saved face
save_lock = threading.Lock()


def maybe_save(visitor_id, name, face_rgb, face):
    now = time.monotonic()
    with save_lock:
        if now - last_save.get(visitor_id, 0.0) < SAVE_INTERVAL:
            return
        last_save[visitor_id] = now
    try:
        save_queue.put_nowait((visitor_id, name, face_rgb.copy(), face, datetime.now()))
    except queue.Full:
        pass  # disk is falling behind; skip this face rather than block


# ---------- inference ----------

def find_faces(image):
    """Return (x, y, w, h) boxes for the faces in an RGB array, biggest first."""
    height, width = image.shape[:2]
    scale = min(1.0, DETECT_WIDTH / width)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    min_side = max(24, int(60 * scale))
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    boxes = []
    for x, y, w, h in faces[:5]:
        x, y, w, h = (int(v / scale) for v in (x, y, w, h))
        # Give the models a little more than just the tight face box.
        pad = int(w * 0.15)
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(width, x + w + pad), min(height, y + h + pad)
        boxes.append([left, top, right - left, bottom - top])
    return boxes


def to_batch(faces, size, mean=None, std=None):
    batch = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in faces]).astype(np.float32) / 255.0
    if mean is not None:
        batch = (batch - mean) / std
    return torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(DEVICE)


def effnet_probs(faces):
    with torch.inference_mode():
        return torch.softmax(effnet(to_batch(faces, effnet_size, effnet_mean, effnet_std)), dim=1).cpu().numpy()


def yolo_probs(faces):
    """YOLO was trained on face crops, so use the class scores of its most confident box in each crop."""
    with torch.inference_mode():
        out = yolo(to_batch(faces, yolo_size))
        out = out[0] if isinstance(out, (list, tuple)) else out  # (batch, 4 + classes, anchors)
        cls = out[:, 4:, :]
        best = cls.max(dim=1).values.argmax(dim=1)
        scores = cls[torch.arange(len(faces)), :, best].cpu().numpy()
    return scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-6, None)


def summarize(p):
    i = int(p.argmax())
    return class_names[i], round(float(p[i]), 4)


def predict_emotions(faces):
    p_eff, p_yolo = effnet_probs(faces), yolo_probs(faces)
    combined = EFFNET_WEIGHT * p_eff + (1 - EFFNET_WEIGHT) * p_yolo

    results = []
    for pc, pe, py in zip(combined, p_eff, p_yolo):
        emotion, confidence = summarize(pc)
        eff_emotion, eff_conf = summarize(pe)
        yolo_emotion, yolo_conf = summarize(py)
        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "scores": {name: round(float(pc[i]), 4) for i, name in enumerate(class_names)},
            "effnet": {"emotion": eff_emotion, "confidence": eff_conf},
            "yolo": {"emotion": yolo_emotion, "confidence": yolo_conf},
        })
    return results


# ---------- routes ----------

@app.get("/")
def index():
    return render_template("index.html", classes=class_names, name=session.get("name", ""),
                           terms_version=TERMS_VERSION)


@app.post("/api/start")
def start():
    data = request.get_json(silent=True) or {}
    name = " ".join(str(data.get("name", "")).split())[:40]
    if not name:
        return jsonify({"error": "Please enter your name."}), 400
    if data.get("accept_terms") is not True:
        return jsonify({"error": "Please accept the Terms & Conditions to continue."}), 400

    with connect() as db:
        cur = db.execute(
            "INSERT INTO visitors (name, terms_version, terms_accepted_at, user_agent) VALUES (?, ?, ?, ?)",
            (name, TERMS_VERSION, datetime.now().isoformat(timespec="seconds"), request.headers.get("User-Agent", "")[:300]),
        )
    session["visitor_id"] = cur.lastrowid
    session["name"] = name
    return jsonify({"name": name})


@app.post("/predict")
def predict():
    visitor_id, name = session.get("visitor_id"), session.get("name")
    if not visitor_id:
        return jsonify({"error": "Enter your name and accept the terms first."}), 401

    started = time.perf_counter()
    jpeg = request.get_data()
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
    if image is None:
        return jsonify({"error": "Could not read the camera frame."}), 400
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = find_faces(image)
    faces = []
    if boxes:
        crops = [image[y:y + h, x:x + w] for x, y, w, h in boxes]
        for box, result in zip(boxes, predict_emotions(crops)):
            result["box"] = box
            faces.append(result)
        # The biggest face is the visitor in front of the camera.
        maybe_save(visitor_id, name, crops[0], faces[0])

    height, width = image.shape[:2]
    server_ms = round((time.perf_counter() - started) * 1000)
    return jsonify({"width": width, "height": height, "faces": faces, "server_ms": server_ms})


@app.get("/api/history")
def history():
    visitor_id = session.get("visitor_id")
    if not visitor_id:
        return jsonify({"error": "Not signed in."}), 401
    limit = min(request.args.get("limit", 50, type=int), 500)
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM predictions WHERE visitor_id = ? ORDER BY id DESC LIMIT ?", (visitor_id, limit),
        ).fetchall()
    return jsonify([{**dict(r), "scores": json.loads(r["scores"]), "box": json.loads(r["box"]),
                     "url": "/" + r["image_path"]} for r in rows])


@app.get("/media/new_face/<path:filename>")
def face_image(filename):
    # Visitors can only open their own saved faces.
    with connect() as db:
        row = db.execute("SELECT visitor_id FROM predictions WHERE image_path = ?",
                         (f"media/new_face/{filename}",)).fetchone()
    if not row or row["visitor_id"] != session.get("visitor_id"):
        abort(404)
    return send_from_directory(FACE_DIR, filename)


def ssl_files():
    """Create a self-signed certificate once, so the camera also works when the site is opened from another device.

    Browsers only allow the camera on https:// pages or on http://localhost.
    """
    cert_path, key_path = BASE_DIR / "data" / "cert.pem", BASE_DIR / "data" / "key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    import ipaddress
    import socket
    from datetime import timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    ips = {"127.0.0.1"}
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    names = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
    names += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in sorted(ips)]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Live Emotion (local)")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption(),
    ))
    return str(cert_path), str(key_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    use_https = os.environ.get("HTTPS", "1") != "0"
    if use_https:
        print(f"Open https://127.0.0.1:{port} (or https://<this computer's IP>:{port} from another device).")
        print("The certificate is self-signed: click Advanced -> Proceed the first time.")
    app.run(host="0.0.0.0", port=port, threaded=True, ssl_context=ssl_files() if use_https else None)
