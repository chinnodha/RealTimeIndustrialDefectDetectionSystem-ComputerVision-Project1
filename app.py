from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI(
    title="Industrial Defect Detection API",
    description="YOLOv8-based industrial surface defect detection",
    version="1.0"
)

# Load ONNX model
model = YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.onnx")



@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    # Convert bytes to numpy array
    image_array = np.frombuffer(contents, np.uint8)

    # Decode image
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    # Run inference
    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

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
        "detections": detections
    }