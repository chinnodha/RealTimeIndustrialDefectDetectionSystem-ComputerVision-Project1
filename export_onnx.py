from ultralytics import YOLO

model = YOLO("C:\\Users\\ADMIN\\Desktop\\runs\\detect\\runs\\train\\defect_detection\\weights\\best.pt")

model.export(
    format="onnx",
    imgsz=640,
    dynamic=True
)

print("ONNX model exported successfully.")