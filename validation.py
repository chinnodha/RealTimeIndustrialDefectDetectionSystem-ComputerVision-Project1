from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/runs/train/defect_detection/weights/best.pt")

# Validate on the validation dataset
metrics = model.val(data="dataset/data.yaml")

print(f"mAP@50     : {metrics.box.map50*100:.2f}%")
print(f"mAP@50-95  : {metrics.box.map*100:.2f}%")
print(f"Precision  : {metrics.box.mp*100:.2f}%")
print(f"Recall     : {metrics.box.mr*100:.2f}%")