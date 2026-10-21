"""Live facial emotion detection with Flask and an EfficientNet-B0 model."""
import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_efficientnet_b0.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["class_names"]
    size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("imagenet_mean", (0.485, 0.456, 0.406))
    std = checkpoint.get("imagenet_std", (0.229, 0.224, 0.225))

    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(classes))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return net, classes, preprocess


model, class_names, preprocess = load_model()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(f"Model loaded on {DEVICE}")


def find_faces(image):
    """Return (x, y, w, h) boxes for the faces in a PIL image, biggest first."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    width, height = image.size
    boxes = []
    for x, y, w, h in faces[:5]:
        # Give the model a little more than just the tight face box.
        pad = int(w * 0.15)
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(width, x + w + pad), min(height, y + h + pad)
        boxes.append([int(left), int(top), int(right - left), int(bottom - top)])
    return boxes


def predict_emotions(face_images):
    batch = torch.stack([preprocess(face) for face in face_images]).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(batch), dim=1).cpu()

    results = []
    for p in probs:
        scores = {name: round(float(p[i]), 4) for i, name in enumerate(class_names)}
        best = max(scores, key=scores.get)
        results.append({"emotion": best, "confidence": scores[best], "scores": scores})
    return results


@app.get("/")
def index():
    return render_template("index.html", classes=class_names)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    frame = data.get("image", "")
    if "," in frame:
        frame = frame.split(",", 1)[1]  # strip "data:image/jpeg;base64,"

    try:
        image = Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the camera frame."}), 400

    boxes = find_faces(image)
    faces = []
    if boxes:
        crops = [image.crop((x, y, x + w, y + h)) for x, y, w, h in boxes]
        for box, result in zip(boxes, predict_emotions(crops)):
            result["box"] = box
            faces.append(result)

    return jsonify({"width": image.width, "height": image.height, "faces": faces})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
