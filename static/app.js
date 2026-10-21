const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startScreen = document.getElementById("start-screen");
const topEmotion = document.getElementById("top-emotion");
const scoresBox = document.getElementById("scores");

// small canvas used to grab frames before sending them to the server
const frameCanvas = document.createElement("canvas");

// one row per emotion in the corner panel
const rows = {};
for (const name of CLASSES) {
  const row = document.createElement("div");
  row.className = "score-row";
  row.innerHTML = `
    <span class="name">${name}</span>
    <div class="bar"><div class="fill"></div></div>
    <span class="value">0%</span>`;
  scoresBox.appendChild(row);
  rows[name] = row;
}

document.getElementById("start-btn").addEventListener("click", async () => {
  try {
    document.documentElement.requestFullscreen?.();
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    video.srcObject = stream;
    await video.play();
    startScreen.style.display = "none";
    detectLoop();
    renderLoop();
  } catch (err) {
    document.getElementById("error").textContent = "Could not open the camera: " + err.message;
  }
});

// latest server result, and the boxes currently on screen (eased toward the result every frame)
let latest = null;
let shown = [];

function grabFrame() {
  // send a smaller copy of the frame, it's faster and good enough for detection
  const scale = 640 / video.videoWidth;
  frameCanvas.width = 640;
  frameCanvas.height = Math.round(video.videoHeight * scale);
  frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  return new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", 0.75));
}

async function detectLoop() {
  // one request in flight at a time; the next one starts as soon as this one returns
  while (true) {
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: await grabFrame(),
      });
      const result = await response.json();
      if (result.faces) {
        latest = result;
        updatePanel(result.faces[0]);
      }
    } catch (err) {
      console.error(err);
      await new Promise((r) => setTimeout(r, 500));
    }
  }
}

function renderLoop() {
  if (overlay.width !== window.innerWidth || overlay.height !== window.innerHeight) {
    overlay.width = window.innerWidth;
    overlay.height = window.innerHeight;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (latest) drawFaces(latest);
  requestAnimationFrame(renderLoop);
}

function drawFaces(result) {
  // the video is shown with object-fit: cover, so scale and crop the same way
  const scale = Math.max(overlay.width / result.width, overlay.height / result.height);
  const offsetX = (overlay.width - result.width * scale) / 2;
  const offsetY = (overlay.height - result.height * scale) / 2;

  shown = result.faces.map((face, i) => {
    let [x, y, w, h] = face.box;
    x = result.width - x - w; // video is mirrored
    const target = [offsetX + x * scale, offsetY + y * scale, w * scale, h * scale];
    const prev = shown[i];
    // ease toward the newest box so it moves smoothly between predictions
    return prev ? prev.map((v, k) => v + (target[k] - v) * 0.35) : target;
  });

  result.faces.forEach((face, i) => {
    const [left, top, width, height] = shown[i];

    ctx.strokeStyle = "#47b300";
    ctx.lineWidth = 3;
    ctx.strokeRect(left, top, width, height);

    // label above the head
    const label = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
    ctx.font = "bold 20px Arial";
    const textWidth = ctx.measureText(label).width;
    const labelX = left + width / 2 - textWidth / 2 - 10;
    const labelY = Math.max(0, top - 40);

    ctx.fillStyle = "#47b300";
    ctx.fillRect(labelX, labelY, textWidth + 20, 32);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, labelX + 10, labelY + 23);
  });
}

function updatePanel(face) {
  if (!face) {
    topEmotion.textContent = "No face";
    for (const name of CLASSES) setRow(name, 0, false);
    return;
  }

  topEmotion.textContent = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
  for (const name of CLASSES) {
    setRow(name, face.scores[name], name === face.emotion);
  }
}

function setRow(name, score, isBest) {
  const row = rows[name];
  const percent = (score * 100).toFixed(1);
  row.querySelector(".fill").style.width = percent + "%";
  row.querySelector(".value").textContent = percent + "%";
  row.classList.toggle("best", isBest);
}
