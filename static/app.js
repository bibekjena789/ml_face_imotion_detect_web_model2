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
  } catch (err) {
    document.getElementById("error").textContent = "Could not open the camera: " + err.message;
  }
});

async function detectLoop() {
  // send a smaller copy of the frame, it's faster and good enough for detection
  const scale = 640 / video.videoWidth;
  frameCanvas.width = 640;
  frameCanvas.height = video.videoHeight * scale;
  frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: frameCanvas.toDataURL("image/jpeg", 0.8) }),
    });
    const result = await response.json();
    drawFaces(result);
    updatePanel(result.faces[0]);
  } catch (err) {
    console.error(err);
  }

  setTimeout(detectLoop, 50);
}

function drawFaces(result) {
  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // the video is shown with object-fit: cover, so scale and crop the same way
  const scale = Math.max(overlay.width / result.width, overlay.height / result.height);
  const offsetX = (overlay.width - result.width * scale) / 2;
  const offsetY = (overlay.height - result.height * scale) / 2;

  for (const face of result.faces) {
    let [x, y, w, h] = face.box;
    x = result.width - x - w; // video is mirrored

    const left = offsetX + x * scale;
    const top = offsetY + y * scale;
    const width = w * scale;
    const height = h * scale;

    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 3;
    ctx.strokeRect(left, top, width, height);

    // label above the head
    const label = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
    ctx.font = "bold 20px Arial";
    const textWidth = ctx.measureText(label).width;
    const labelX = left + width / 2 - textWidth / 2 - 10;
    const labelY = Math.max(0, top - 40);

    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(labelX, labelY, textWidth + 20, 32);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, labelX + 10, labelY + 23);
  }
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
