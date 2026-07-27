import cv2
import time
import argparse
import os
from ultralytics import YOLO

# ==========================
# Load YOLO Model
# ==========================
MODEL_PATH = os.path.join(
    "runs",
    "detect",
    "runs",
    "first_test",
    "weights",
    "best.pt"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# Supported image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==========================
# Check if input is image
# ==========================
def is_image_file(path):
    return isinstance(path, str) and path.lower().endswith(IMAGE_EXTS)


# ==========================
# Open webcam/video
# ==========================
def open_video_source(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    return cap


# ==========================
# Image Detection
# ==========================
def run_image(image_path, display=True):

    print(f"\nReading image: {image_path}")

    frame = cv2.imread(image_path)

    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    print("Image loaded successfully.")

    print("Running YOLO prediction...")

    results = model.predict(
        source=frame,
        conf=0.25,
        save=False,
        verbose=True
    )

    annotated = results[0].plot()

    os.makedirs("results", exist_ok=True)

    output_path = os.path.join(
        "results",
        os.path.splitext(os.path.basename(image_path))[0] + "_result.jpg"
    )

    cv2.imwrite(output_path, annotated)

    print(f"\nResult saved to:")
    print(output_path)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("\nNo defects detected.")
    else:
        print(f"\nDetected {len(boxes)} defect(s):")

        names = model.names

        for i, box in enumerate(boxes):

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            print(
                f"{i+1}. {names[cls]}  | Confidence: {conf:.2f}"
            )

    if display:
        cv2.imshow("Industrial Defect Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==========================
# Webcam / Video Detection
# ==========================
def run_video(source, display=True, max_frames=None):

    cap = open_video_source(source)

    start = time.time()
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        annotated = results[0].plot()

        fps = frame_count / max(time.time() - start, 1e-6)

        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if display:

            cv2.imshow(
                "Industrial Defect Detection",
                annotated
            )

            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

        if max_frames is not None:

            if frame_count >= max_frames:
                break

    cap.release()

    cv2.destroyAllWindows()


# ==========================
# Main
# ==========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Industrial Defect Detection"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="0 = Webcam | Image Path | Video Path"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None
    )

    args = parser.parse_args()

    source = (
        int(args.source)
        if str(args.source).isdigit()
        else args.source
    )

    print("\n====================================")
    print("Industrial Defect Detection Started")
    print("====================================")

    print(f"Source: {source}")

    if is_image_file(source):

        print("Mode: Image")

        run_image(
            source,
            display=not args.no_display
        )

    else:

        print("Mode: Webcam / Video")

        run_video(
            source,
            display=not args.no_display,
            max_frames=args.max_frames
        )

    print("\nProgram Finished Successfully!")