from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        device="cpu",

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        patience=30,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        project="runs/tuning",
        name="defect_detection_tuned",
        save=True,
        verbose=True
    )

if __name__ == "__main__":
    main()