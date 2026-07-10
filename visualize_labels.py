import os
import cv2

# ======================================================
# Dataset Paths
# ======================================================

IMAGE_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

# ======================================================
# Class Names
# ======================================================

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ======================================================
# Read all images
# ======================================================

image_files = sorted(os.listdir(IMAGE_DIR))

print(f"Found {len(image_files)} images.\n")

# ======================================================
# Display images one by one
# ======================================================

for image_name in image_files:

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    # Skip if label doesn't exist
    if not os.path.exists(label_path):
        print(f"Label not found for {image_name}")
        continue

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not open {image_name}")
        continue

    img_height, img_width = image.shape[:2]

    # Read label file
    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            box_width = float(values[3])
            box_height = float(values[4])

            # Convert YOLO coordinates to pixel coordinates
            xmin = int((x_center - box_width / 2) * img_width)
            ymin = int((y_center - box_height / 2) * img_height)

            xmax = int((x_center + box_width / 2) * img_width)
            ymax = int((y_center + box_height / 2) * img_height)

            # Draw bounding box
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2
            )

            # Draw class name
            cv2.putText(
                image,
                classes[class_id],
                (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # Display image name
    cv2.putText(
        image,
        image_name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Resize image for easier viewing (optional)
    display = cv2.resize(image, (800, 600))

    cv2.imshow("YOLO Label Verification", display)

    print(f"Showing: {image_name}")

    key = cv2.waitKey(0)

    # ESC key exits the program
    if key == 27:
        break

cv2.destroyAllWindows()