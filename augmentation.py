import os
import cv2
import albumentations as A

# ==========================================================
# Dataset Paths
# ==========================================================

IMAGE_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

# Number of augmented copies to generate per image
NUM_AUGMENTATIONS = 2

# ==========================================================
# Albumentations Pipeline
# ==========================================================

transform = A.Compose(
    [
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3)
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"]
    )
)

# ==========================================================
# Read all images
# ==========================================================

images = os.listdir(IMAGE_DIR)

print(f"Found {len(images)} images")

# ==========================================================
# Process each image
# ==========================================================

for image_name in images:

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)

    label_path = os.path.join(
        LABEL_DIR,
        image_name.rsplit(".", 1)[0] + ".txt"
    )

    # Skip if label file doesn't exist
    if not os.path.exists(label_path):
        print(f"Label not found for {image_name}")
        continue

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read {image_name}")
        continue

    # ======================================================
    # Read YOLO labels
    # ======================================================

    bboxes = []
    class_labels = []

    with open(label_path, "r") as f:

        for line in f.readlines():

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])

            bboxes.append([x, y, w, h])
            class_labels.append(class_id)

    # ======================================================
    # Generate augmented images
    # ======================================================

    for i in range(NUM_AUGMENTATIONS):

        augmented = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        aug_image = augmented["image"]
        aug_boxes = augmented["bboxes"]
        aug_classes = augmented["class_labels"]

        # Save image

        new_image_name = (
            image_name.rsplit(".", 1)[0]
            + f"_aug{i+1}.jpg"
        )

        new_image_path = os.path.join(
            IMAGE_DIR,
            new_image_name
        )

        cv2.imwrite(new_image_path, aug_image)

        # Save labels

        new_label_name = (
            image_name.rsplit(".", 1)[0]
            + f"_aug{i+1}.txt"
        )

        new_label_path = os.path.join(
            LABEL_DIR,
            new_label_name
        )

        with open(new_label_path, "w") as f:

            for cls, box in zip(aug_classes, aug_boxes):

                f.write(
                    f"{cls} "
                    f"{box[0]:.6f} "
                    f"{box[1]:.6f} "
                    f"{box[2]:.6f} "
                    f"{box[3]:.6f}\n"
                )

print("\nData augmentation completed successfully!")