from ultralytics import YOLO

# Load ONNX model
model = YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.onnx")   # Update path if needed

# Predict on an image
results = model.predict(
    source=r"test_images\sample.jpg",
    conf=0.25,
    save=True,
    show=True
)

print("Prediction completed using ONNX model!")