import os
from ultralytics import YOLO

# -----------------------------
# Load trained model
# -----------------------------
model = YOLO(r"C:\Users\ADMIN\Desktop\runs\detect\runs\train\defect_detection\weights\best.pt")

images_path = r"dataset/images/validation"
labels_path = r"dataset/labels/validation"

IOU_THRESHOLD = 0.5
CONF = 0.25


# -----------------------------
# IoU Function
# -----------------------------
def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


TP = 0
FP = 0
FN = 0

for image in os.listdir(images_path):

    if not image.endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(images_path, image)

    label_file = os.path.join(
        labels_path,
        os.path.splitext(image)[0] + ".txt"
    )

    # -----------------------------
    # Read Ground Truth
    # -----------------------------
    gt_boxes = []

    if os.path.exists(label_file):

        with open(label_file) as f:

            for line in f:

                cls, x, y, w, h = map(float, line.split())

                xmin = x - w/2
                ymin = y - h/2
                xmax = x + w/2
                ymax = y + h/2

                gt_boxes.append(
                    (int(cls), [xmin, ymin, xmax, ymax])
                )

    # -----------------------------
    # Prediction
    # -----------------------------
    results = model.predict(
        img_path,
        conf=CONF,
        verbose=False
    )

    pred_boxes = []

    for box in results[0].boxes:

        cls = int(box.cls)

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # normalize
        h_img, w_img = results[0].orig_shape

        x1 /= w_img
        x2 /= w_img
        y1 /= h_img
        y2 /= h_img

        pred_boxes.append(
            (cls, [x1, y1, x2, y2])
        )

    matched_gt = set()

    for pred_cls, pred_box in pred_boxes:

        matched = False

        for idx, (gt_cls, gt_box) in enumerate(gt_boxes):

            if idx in matched_gt:
                continue

            if pred_cls != gt_cls:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou >= IOU_THRESHOLD:

                TP += 1
                matched_gt.add(idx)
                matched = True
                break

        if not matched:
            FP += 1

    FN += len(gt_boxes) - len(matched_gt)


precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)

print("\n==============================")
print("ERROR ANALYSIS REPORT")
print("==============================")
print(f"True Positives : {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision      : {precision*100:.2f}%")
print(f"Recall         : {recall*100:.2f}%")
print("==============================")