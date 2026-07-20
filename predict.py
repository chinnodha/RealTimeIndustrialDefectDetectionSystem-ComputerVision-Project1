from ultralytics import YOLO

# Load the trained model
model = YOLO("C:\\Users\\ADMIN\\Desktop\\runs\\detect\\runs\\train\\defect_detection\\weights\\best.pt")

# Predict on an image
results = model.predict(
    source="test_images/sample_3.jpg",
    conf=0.2,
    save=True,
    show=True
)

print("Prediction completed!")