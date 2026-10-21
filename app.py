# -*- coding: utf-8 -*-
"""Responsive Flask web app for the Kaggle EfficientNet-B0 emotion checkpoint."""
import os, io, base64
from pathlib import Path

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


def predict_image(image):
    if model is None or preprocess is None:
        raise RuntimeError(model_error or "Model is not loaded.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax().item())
    return {"emotion": class_names[idx], "confidence": round(float(probs[idx]), 6),
            "probabilities": {name: round(float(probs[i]), 6) for i, name in enumerate(class_names)},
            "device": str(DEVICE),
            "note": "Model estimate only; it does not reliably establish a person's inner emotional state."}


@app.get("/")
def index():
    return render_template("index.html", model_ready=model is not None, model_error=model_error)


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
            data = request.json["image"]
            if "," in data and data.split(",", 1)[0].startswith("data:"):
                data = data.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
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


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"Image too large. Maximum is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    # Development only. Use HTTPS + a production WSGI server for deployment.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
