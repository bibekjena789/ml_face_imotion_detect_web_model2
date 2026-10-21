# Live Emotion

A small Flask app that shows your camera across the whole browser window with your facial emotion live.
Visitors enter their name and accept the Terms & Conditions, then a label with their name and emotion
appears above their face. The corner panel shows the score for every emotion and the response time.

## Setup

Put both trained models in `models/`: `emotion_efficientnet_b0.pth` and `best.pt` (YOLOv8), then:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
# ultralytics pulls in opencv-python; if it clashes with opencv-python-headless:
# pip install --no-deps ultralytics; pip install matplotlib psutil pyyaml requests polars ultralytics-thop
python app.py
```

Open http://127.0.0.1:5000, enter your name, accept the terms and click **Go**.

Notes:
- The camera only works on `localhost` or over HTTPS.
- Use `opencv-python-headless` 4.x. OpenCV 5 no longer ships the face detector file this app uses.
- The model can be wrong. A facial expression doesn't always show what someone actually feels.

## How predictions work

Faces are found with OpenCV, then each face goes through both models and their scores are averaged
(`EFFNET_WEIGHT`, default `0.5`, sets EfficientNet's share). A frame takes about 70 ms on CPU,
well under the 500 ms target. The panel shows the time in red if a frame takes longer than 500 ms.

## Saved faces

While a face is in view, the visitor's face is saved about once per second (`SAVE_INTERVAL`, in seconds).

- Images: `media/new_face/<name>_<YYYY-MM-DD_HH-MM-SS-mmm>.jpg`
- Database: `data/emotions.db` (SQLite)
  - `visitors`: name, accepted terms version and time
  - `predictions`: image path, combined emotion and scores, each model's own prediction, face box, time
- `GET /api/history?limit=50` returns the signed-in visitor's own saved predictions.

`media/` and `data/` are created on first run and are ignored by git.
