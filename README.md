# Live Emotion

A small Flask app that opens your camera full screen and shows your facial emotion live.
A label appears above each face, and the corner panel shows the score for every emotion.

## Setup

Put the trained model at `models/emotion_efficientnet_b0.pth`, then:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and click **Start camera**.

Notes:
- The camera only works on `localhost` or over HTTPS.
- Use `opencv-python-headless` 4.x. OpenCV 5 no longer ships the face detector file this app uses.
- The model can be wrong. A facial expression doesn't always show what someone actually feels.
