from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,          # Smaller batch for CPU
        workers=2,        # Good choice for Windows
        device="cpu",
        project="runs/train",
        name="defect_detection",
        save=True,
        patience=20,
        verbose=True
    )

if __name__ == "__main__":
    main()