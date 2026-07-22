import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.onnx")

cap = cv2.VideoCapture("test_video.mp4")

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width,height)
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=0.25, verbose=False)

    annotated = results[0].plot()

    out.write(annotated)

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()