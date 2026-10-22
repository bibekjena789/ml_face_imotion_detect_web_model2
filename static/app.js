const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startScreen = document.getElementById("start-screen");
const nameInput = document.getElementById("name-input");
const acceptInput = document.getElementById("accept-input");
const startBtn = document.getElementById("start-btn");
const errorBox = document.getElementById("error");
const userName = document.getElementById("user-name");
const topEmotion = document.getElementById("top-emotion");
const scoresBox = document.getElementById("scores");
const modelsBox = document.getElementById("models");
const latencyBox = document.getElementById("latency");

const LATENCY_TARGET_MS = 700;

// small canvas used to grab frames before sending them to the server
const frameCanvas = document.createElement("canvas");

let visitorName = "";
let running = false;

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

// Go only works once there's a name and the terms are accepted
function updateStartButton() {
  startBtn.disabled = !(nameInput.value.trim() && acceptInput.checked);
}
nameInput.addEventListener("input", updateStartButton);
acceptInput.addEventListener("change", updateStartButton);
updateStartButton();

startScreen.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";
  startBtn.disabled = true;
  try {
    const response = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: nameInput.value, accept_terms: acceptInput.checked }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    visitorName = data.name;
    userName.textContent = visitorName;

    if (!navigator.mediaDevices?.getUserMedia) {
      // browsers hide the camera on plain http:// pages, except on localhost
      const secureUrl = location.href.replace(/^http:/, "https:");
      throw new Error(window.isSecureContext
        ? "This browser doesn't support camera access."
        : `The camera only works over https. Open ${secureUrl} instead.`);
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    video.srcObject = stream;
    await video.play();
    startScreen.style.display = "none";
    running = true;
    detectLoop();
    renderLoop();
  } catch (err) {
    errorBox.textContent = err.name === "NotAllowedError" || err.name === "NotFoundError"
      ? "Could not open the camera: " + err.message
      : err.message;
    updateStartButton();
  }
});

document.getElementById("exit-btn").addEventListener("click", () => {
  running = false;
  video.srcObject?.getTracks().forEach((track) => track.stop());
  location.reload();
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
  return new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", 0.8));
}

async function detectLoop() {
  // one request in flight at a time; the next one starts as soon as this one returns
  while (running) {
    const started = performance.now();
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: await grabFrame(),
        signal: AbortSignal.timeout(2000),
      });
      const result = await response.json();
      if (response.status === 401) {
        location.reload(); // session expired, ask for the name again
        return;
      }
      if (result.faces) {
        latest = result;
        updatePanel(result.faces[0]);
        showLatency(performance.now() - started, result.server_ms);
      }
    } catch (err) {
      console.error(err);
      await new Promise((r) => setTimeout(r, 300));
    }
  }
}

function showLatency(totalMs, serverMs) {
  latencyBox.textContent = `${Math.round(totalMs)} ms total · ${serverMs} ms server`;
  latencyBox.classList.toggle("slow", totalMs > LATENCY_TARGET_MS);
}

function renderLoop() {
  if (!running) return;
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

    // label above the head; the biggest face is the visitor, so it gets their name
    const emotion = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
    const label = i === 0 ? `${visitorName} · ${emotion}` : emotion;
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
    modelsBox.textContent = "";
    for (const name of CLASSES) setRow(name, 0, false);
    return;
  }

  topEmotion.textContent = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
  modelsBox.textContent =
    `EfficientNet: ${face.effnet.emotion} ${Math.round(face.effnet.confidence * 100)}% · ` +
    `YOLO: ${face.yolo.emotion} ${Math.round(face.yolo.confidence * 100)}%`;
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
