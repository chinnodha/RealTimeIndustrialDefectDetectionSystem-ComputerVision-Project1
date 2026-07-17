from ultralytics import YOLO

# Load the trained model
model = YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\tuning\defect_detection_tuned-2\weights\best.pt")
model1=YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.pt")
# Validate on the validation dataset
metrics = model.val(data=r"dataset\data.yaml")
metrics1 = model1.val(data=r"dataset\data.yaml")

print("Tuned Model:")
print(f"mAP@50     : {metrics.box.map50*100:.2f}%")
print(f"mAP@50-95  : {metrics.box.map*100:.2f}%")
print(f"Precision  : {metrics.box.mp*100:.2f}%")
print(f"Recall     : {metrics.box.mr*100:.2f}%")

print("\nOriginal Model:")
print(f"mAP@50     : {metrics1.box.map50*100:.2f}%")
print(f"mAP@50-95  : {metrics1.box.map*100:.2f}%")
print(f"Precision  : {metrics1.box.mp*100:.2f}%")
print(f"Recall     : {metrics1.box.mr*100:.2f}%")