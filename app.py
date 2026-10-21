# -*- coding: utf-8 -*-
"""Responsive Flask web app for the Kaggle EfficientNet-B0 emotion checkpoint."""
import os, io, base64
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("EMOTION_MODEL_PATH", str(BASE_DIR / "models" / "emotion_efficientnet_b0.pth")))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
model = None
class_names = DEFAULT_CLASSES
image_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
preprocess = None
model_error = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("[FACE] Haar cascade failed to load; whole frames will be classified.")
    face_cascade = None


def load_model():
    global model, class_names, image_size, mean, std, preprocess, model_error
    if not MODEL_PATH.is_file():
        model_error = f"Checkpoint not found: {MODEL_PATH}. Copy emotion_efficientnet_b0.pth into the models folder."
        print("[MODEL]", model_error)
        return
    try:
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
        class_names = ckpt.get("class_names", DEFAULT_CLASSES)
        image_size = int(ckpt.get("image_size", 224))
        mean = tuple(ckpt.get("imagenet_mean", mean))
        std = tuple(ckpt.get("imagenet_std", std))
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(class_names))
        net.load_state_dict(ckpt["model_state_dict"], strict=True)
        model = net.to(DEVICE).eval()
        preprocess = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        model_error = None
        print(f"[MODEL] Loaded {MODEL_PATH} on {DEVICE}")
    except Exception as exc:
        model_error = f"Model load failed: {type(exc).__name__}: {exc}"
        print("[MODEL]", model_error)


load_model()


def decode_image(data):
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data, validate=True)))


def detect_faces(image, max_faces=5):
    """Return face boxes (x, y, w, h) in image pixels, largest first, padded for context."""
    if face_cascade is None:
        return []
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(40, min(gray.shape) // 10)
    found = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
    W, H = image.size
    boxes = []
    for x, y, w, h in sorted((tuple(int(v) for v in f) for f in found), key=lambda b: b[2] * b[3], reverse=True)[:max_faces]:
        pad = int(0.15 * w)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def classify(crops):
    if model is None or preprocess is None:
        raise RuntimeError(model_error or "Model is not loaded.")
    x = torch.stack([preprocess(c) for c in crops]).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1).cpu()
    results = []
    for p in probs:
        idx = int(p.argmax().item())
        results.append({"emotion": class_names[idx], "confidence": round(float(p[idx]), 6),
                        "probabilities": {name: round(float(p[i]), 6) for i, name in enumerate(class_names)}})
    return results


def predict_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    boxes = detect_faces(image, max_faces=1)
    crop = image.crop((boxes[0][0], boxes[0][1], boxes[0][0] + boxes[0][2], boxes[0][1] + boxes[0][3])) if boxes else image
    result = classify([crop])[0]
    result.update({"face_box": list(boxes[0]) if boxes else None, "face_found": bool(boxes), "device": str(DEVICE),
                   "note": "Model estimate only; it does not reliably establish a person's inner emotional state."})
    return result


def predict_faces(image):
    image = image.convert("RGB")
    boxes = detect_faces(image)
    faces = []
    if boxes:
        crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
        for box, res in zip(boxes, classify(crops)):
            res["box"] = list(box)
            faces.append(res)
    return {"faces": faces, "width": image.size[0], "height": image.size[1], "device": str(DEVICE)}


@app.get("/")
def index():
    return render_template("index.html", model_ready=model is not None, model_error=model_error)


@app.get("/live")
def live_page():
    return render_template("live.html", model_ready=model is not None, model_error=model_error, classes=class_names)


@app.get("/health")
def health():
    return jsonify({"ok": True, "model_ready": model is not None, "device": str(DEVICE), "model_path": str(MODEL_PATH), "error": model_error, "classes": class_names}), (200 if model is not None else 503)


@app.post("/api/predict")
def api_predict():
    if model is None:
        return jsonify({"error": model_error or "Model not loaded"}), 503
    try:
        if "image" in request.files:
            file = request.files["image"]
            if not file.filename:
                return jsonify({"error": "Choose an image first."}), 400
            image = Image.open(file.stream)
        elif request.is_json and request.json and request.json.get("image"):
            image = decode_image(request.json["image"])
        else:
            return jsonify({"error": "Send an image using the image field."}), 400
        return jsonify(predict_image(image))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return jsonify({"error": f"Invalid image: {exc}"}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {type(exc).__name__}: {exc}"}), 500


@app.post("/api/predict_live")
def api_predict_live():
    """Detect every face in a live camera frame and classify each one."""
    if model is None:
        return jsonify({"error": model_error or "Model not loaded"}), 503
    data = request.get_json(silent=True) or {}
    if not data.get("image"):
        return jsonify({"error": "Send a frame as a base64 data URL in the image field."}), 400
    try:
        return jsonify(predict_faces(decode_image(data["image"])))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return jsonify({"error": f"Invalid frame: {exc}"}), 400
    except Exception as exc:
        app.logger.exception("Live prediction failed")
        return jsonify({"error": f"Live prediction failed: {type(exc).__name__}: {exc}"}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"Image too large. Maximum is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    # Development only. Use HTTPS + a production WSGI server for deployment.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
