from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import numpy as np
import cv2
import time

app = FastAPI(
    title="Industrial Defect Detection API",
    description="YOLOv8-based industrial surface defect detection",
    version="1.0"
)

# Load ONNX model
model = YOLO(
    r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.onnx"
)


# ==========================================================
# PROMETHEUS METRICS
# ==========================================================

REQUEST_COUNT = Counter(
    "defect_api_requests_total",
    "Total number of prediction requests"
)

INFERENCE_LATENCY = Histogram(
    "defect_model_inference_latency_seconds",
    "Model inference latency in seconds"
)

API_UPTIME = Gauge(
    "defect_api_uptime",
    "API uptime status"
)

API_UPTIME.set(1)


# Prometheus endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running"
    }


# ==========================================================
# PREDICTION
# ==========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    REQUEST_COUNT.inc()

    # Read uploaded image
    contents = await file.read()

    # Convert bytes to numpy array
    image_array = np.frombuffer(contents, np.uint8)

    # Decode image
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    # Start timer
    start_time = time.perf_counter()

    # Run inference
    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    # End timer
    inference_time = time.perf_counter() - start_time

    # Record latency
    INFERENCE_LATENCY.observe(inference_time)

    result = results[0]

    detections = []

    for box in result.boxes:

        cls = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[cls],
            "confidence": round(confidence, 3),
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        })

    return {
        "filename": file.filename,
        "inference_time_seconds": round(inference_time, 4),
        "detections": detections
    }