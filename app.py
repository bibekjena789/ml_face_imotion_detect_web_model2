"""Live facial emotion detection with Flask and an EfficientNet-B0 model."""
import json
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, send_from_directory
from torchvision import models

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_efficientnet_b0.pth"
DB_PATH = BASE_DIR / "data" / "emotions.db"
CAPTURE_DIR = BASE_DIR / "captures"
SAVE_INTERVAL = float(os.environ.get("SAVE_INTERVAL", 1.0))  # seconds between saved frames
DETECT_WIDTH = 320  # face detection runs on a smaller copy of the frame
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["class_names"]
    size = int(checkpoint.get("image_size", 224))
    mean = np.array(checkpoint.get("imagenet_mean", (0.485, 0.456, 0.406)), dtype=np.float32)
    std = np.array(checkpoint.get("imagenet_std", (0.229, 0.224, 0.225)), dtype=np.float32)

    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(classes))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(DEVICE).eval()

    # The first forward pass is much slower than the rest, so do it now instead of on the first request.
    with torch.inference_mode():
        net(torch.zeros(1, 3, size, size, device=DEVICE))
    return net, classes, size, mean, std


model, class_names, image_size, mean, std = load_model()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(f"Model loaded on {DEVICE}")


# ---------- storage ----------

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    CAPTURE_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS captures (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT,
                emotion    TEXT NOT NULL,
                confidence REAL NOT NULL,
                faces      TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")


def save_worker():
    """Write frames to disk and the database off the request thread, so saving never slows predictions."""
    db = sqlite3.connect(DB_PATH)
    while True:
        jpeg, faces, created_at = save_queue.get()
        try:
            best = faces[0]
            cur = db.execute(
                "INSERT INTO captures (emotion, confidence, faces, created_at) VALUES (?, ?, ?, ?)",
                (best["emotion"], best["confidence"], json.dumps(faces), created_at),
            )
            # Name the file after the row id so images are numbered in order: 000001.jpg, 000002.jpg, ...
            filename = f"{cur.lastrowid:06d}.jpg"
            (CAPTURE_DIR / filename).write_bytes(jpeg)
            db.execute("UPDATE captures SET filename = ? WHERE id = ?", (filename, cur.lastrowid))
            db.commit()
        except Exception as err:
            db.rollback()
            print(f"Could not save capture: {err}")


init_db()
save_queue = queue.Queue(maxsize=50)
threading.Thread(target=save_worker, daemon=True).start()
last_save = 0.0
save_lock = threading.Lock()


def maybe_save(jpeg, faces):
    global last_save
    now = time.monotonic()
    with save_lock:
        if now - last_save < SAVE_INTERVAL:
            return
        last_save = now
    try:
        save_queue.put_nowait((jpeg, faces, datetime.now().isoformat(timespec="milliseconds")))
    except queue.Full:
        pass  # disk is falling behind; skip this frame rather than block


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
        # Give the model a little more than just the tight face box.
        pad = int(w * 0.15)
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(width, x + w + pad), min(height, y + h + pad)
        boxes.append([left, top, right - left, bottom - top])
    return boxes


def preprocess(face):
    face = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)
    face = (face.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(face.transpose(2, 0, 1))


def predict_emotions(face_images):
    batch = torch.stack([preprocess(face) for face in face_images]).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()

    results = []
    for p in probs:
        scores = {name: round(float(p[i]), 4) for i, name in enumerate(class_names)}
        best = max(scores, key=scores.get)
        results.append({"emotion": best, "confidence": scores[best], "scores": scores})
    return results


# ---------- routes ----------

@app.get("/")
def index():
    return render_template("index.html", classes=class_names)


@app.post("/predict")
def predict():
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
        maybe_save(jpeg, faces)

    height, width = image.shape[:2]
    return jsonify({"width": width, "height": height, "faces": faces})


@app.get("/captures")
def list_captures():
    limit = min(request.args.get("limit", 50, type=int), 500)
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT id, filename, emotion, confidence, faces, created_at FROM captures "
            "WHERE filename IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    return jsonify([{**dict(r), "faces": json.loads(r["faces"]), "url": f"/captures/{r['filename']}"} for r in rows])


@app.get("/captures/<path:filename>")
def capture_image(filename):
    return send_from_directory(CAPTURE_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
