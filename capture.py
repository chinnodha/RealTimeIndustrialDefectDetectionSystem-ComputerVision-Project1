import cv2
import time
import argparse
from ultralytics import YOLO


def open_video_source(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def run_capture_loop(source, model_path=None, display=True, max_frames=None):
    cap = open_video_source(source)

    model = YOLO(model_path) if model_path else None

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame could not be read.")
            break

        frame_count += 1

        if model is not None:
            results = model(frame, verbose=False)
            frame = results[0].plot()

        if display:
            cv2.imshow("Defect Detection Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if max_frames and frame_count >= max_frames:
            break

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video capture + inference module for defect detection")
    parser.add_argument("--source", default=0, help="Video source: 0 for webcam, or a file/stream path")
    parser.add_argument("--model", default=None, help="Path to trained YOLOv8 model (best.pt)")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for testing)")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    run_capture_loop(source, model_path=args.model, display=not args.no_display, max_frames=args.max_frames)