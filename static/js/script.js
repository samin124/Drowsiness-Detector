const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusText = document.getElementById('status');
const detailsText = document.getElementById('details');
const alarm = document.getElementById('alarm');

const IMG_SIZE = 224;
canvasElement.width = IMG_SIZE;
canvasElement.height = IMG_SIZE;

let alarmPlaying = false;

// MediaPipe Face Mesh setup
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceMesh.send({image: videoElement});
  },
  width: 640,
  height: 480
});
camera.start();

// Landmark indices for eyes and mouth (MediaPipe)
const LEFT_EYE_INDICES = [33, 133, 160, 158, 159, 144, 153, 154, 155, 133];
const RIGHT_EYE_INDICES = [362, 263, 387, 385, 386, 373, 380, 381, 382, 263];
const MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324];

// Offscreen canvas for cropping and resizing
const offCanvas = document.createElement('canvas');
offCanvas.width = IMG_SIZE;
offCanvas.height = IMG_SIZE;
const offCtx = offCanvas.getContext('2d');

function onResults(results) {
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    statusText.textContent = "No face detected";
    stopAlarm();
    return;
  }

  const landmarks = results.multiFaceLandmarks[0];

  // Get bounding boxes with padding
  const leftEyeCoords = getRegionBoundingBox(landmarks, LEFT_EYE_INDICES);
  const rightEyeCoords = getRegionBoundingBox(landmarks, RIGHT_EYE_INDICES);
  const mouthCoords = getRegionBoundingBox(landmarks, MOUTH_INDICES);

  // Crop base64 images
  const leftEyeCrop = cropRegion(leftEyeCoords);
  const rightEyeCrop = cropRegion(rightEyeCoords);
  const mouthCrop = cropRegion(mouthCoords);

  // Send crops for prediction
  predictDrowsiness(leftEyeCrop, rightEyeCrop, mouthCrop);
}

function getRegionBoundingBox(landmarks, indices) {
  let xs = [], ys = [];
  indices.forEach(i => {
    xs.push(landmarks[i].x);
    ys.push(landmarks[i].y);
  });

  const xMinRaw = Math.min(...xs) * videoElement.videoWidth;
  const xMaxRaw = Math.max(...xs) * videoElement.videoWidth;
  const yMinRaw = Math.min(...ys) * videoElement.videoHeight;
  const yMaxRaw = Math.max(...ys) * videoElement.videoHeight;

  const widthRaw = xMaxRaw - xMinRaw;
  const heightRaw = yMaxRaw - yMinRaw;

  const paddingFactor = 0.3;

  const widthPadded = widthRaw * (1 + paddingFactor);
  const heightPadded = heightRaw * (1 + paddingFactor);

  const centerX = (xMinRaw + xMaxRaw) / 2;
  const centerY = (yMinRaw + yMaxRaw) / 2;

  const xMin = Math.max(0, centerX - widthPadded / 2);
  const yMin = Math.max(0, centerY - heightPadded / 2);

  const xMax = Math.min(videoElement.videoWidth, centerX + widthPadded / 2);
  const yMax = Math.min(videoElement.videoHeight, centerY + heightPadded / 2);

  const width = xMax - xMin;
  const height = yMax - yMin;

  return { xMin, yMin, width, height };
}

function cropRegion(coords) {
  const { xMin, yMin, width, height } = coords;

  offCtx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  offCtx.drawImage(
    videoElement,
    xMin, yMin, width, height,
    0, 0, IMG_SIZE, IMG_SIZE
  );

  return offCanvas.toDataURL('image/jpeg');
}

let lastSentTime = 0;
const MIN_TIME_BETWEEN_REQUESTS = 1500; // ms

async function predictDrowsiness(leftEye, rightEye, mouth) {
  const now = Date.now();
  if (now - lastSentTime < MIN_TIME_BETWEEN_REQUESTS) return;
  lastSentTime = now;

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        left_eye: leftEye,
        right_eye: rightEye,
        mouth: mouth
      })
    });

    const data = await response.json();

    if(data.error){
      statusText.textContent = "Error: " + data.error;
      stopAlarm();
      return;
    }

    statusText.textContent = data.status === "Drowsy" ? "You are DROWSY!" : "You are Alert.";
    detailsText.textContent = `Left Eye: ${data.left_eye}, Right Eye: ${data.right_eye}, Mouth: ${data.mouth}`;

    if(data.status === "Drowsy" && !alarmPlaying){
      alarm.play();
      alarmPlaying = true;
    } else if(data.status === "Alert" && alarmPlaying){
      stopAlarm();
    }
  } catch (err) {
    statusText.textContent = "Prediction failed.";
    console.error(err);
    stopAlarm();
  }
}

function stopAlarm(){
  alarm.pause();
  alarm.currentTime = 0;
  alarmPlaying = false;
}
