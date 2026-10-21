# Face Emotion Lab — Flask camera app

Responsive Flask application for the exported EfficientNet-B0 facial-expression model. Supports desktop webcam, mobile front/rear camera, image upload, eight class scores, `/api/predict`, and `/health`.

## 1. Add your trained checkpoint

Download this file from Kaggle's Output panel:

`/kaggle/working/emotion_model_export/emotion_efficientnet_b0.pth`

Copy it to:

`models/emotion_efficientnet_b0.pth`

The model file is not bundled in this source ZIP; it must be copied from your Kaggle run.

## 2. Install and run (Windows PowerShell)

Python 3.10–3.12 recommended.

```powershell
cd facial_emotion_flask_app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If the default PyTorch install is not suitable for your machine, install a matching build using the official PyTorch installation selector. CPU inference is supported.

Open `http://127.0.0.1:5000` and check `http://127.0.0.1:5000/health`.

## 3. Use a phone camera

Browser camera access normally requires HTTPS, except on `localhost`. To use the camera from a phone, deploy behind HTTPS with a valid certificate or secure tunnel. A phone opening a plain `http://192.168.x.x:5000` address may not be allowed to use the camera. Do not expose Flask's development server directly to the public internet; use a production WSGI server and HTTPS for deployment.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/predict -F "image=@face.jpg"
```

The app does not save images by design. Obtain consent before analyzing another person's image. The model can be wrong; facial expression is not a reliable measure of someone's inner feelings or intent.
"# ml_face_imotion_detect_web_model2" 
