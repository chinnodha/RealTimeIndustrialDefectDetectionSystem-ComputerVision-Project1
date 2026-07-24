import cv2
import time
from ultralytics import YOLO

# Load your ONNX or PyTorch model
# model = YOLO("best.onnx")
# OR
model = YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.onnx")

# Open the video
cap = cv2.VideoCapture("test_video.mp4")

total_time = 0
frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    start = time.time()

    results = model.predict(frame, verbose=False)

    end = time.time()

    inference_time = end - start
    total_time += inference_time
    frame_count += 1

cap.release()

average_time = total_time / frame_count
average_fps = frame_count / total_time

print(f"Frames Processed      : {frame_count}")
print(f"Average Inference Time: {average_time:.4f} seconds")
print(f"Average FPS           : {average_fps:.2f}")