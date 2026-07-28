import cv2
import argparse
from ultralytics import YOLO


def detect_image(image_path, model_path, output_path="output.jpg", display=True):
    model = YOLO(model_path)

    results = model(image_path)
    annotated_frame = results[0].plot()

    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved annotated result to {output_path}")

    if display:
        cv2.imshow("Detection Result", annotated_frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    boxes = results[0].boxes
    print(f"Detections found: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"  - {label} (confidence: {conf:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 defect detection on a single image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", default="best.pt", help="Path to trained YOLOv8 model")
    parser.add_argument("--output", default="output.jpg", help="Path to save the annotated image")
    parser.add_argument("--no-display", action="store_true", help="Don't open a display window")
    args = parser.parse_args()

    detect_image(args.image, args.model, args.output, display=not args.no_display)