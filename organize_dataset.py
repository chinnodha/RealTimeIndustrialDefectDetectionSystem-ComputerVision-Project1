import os
import shutil

# Source folders
TRAIN_SOURCE = r"C:\Users\ADMIN\Downloads\archive (7)\NEU-DET\train\images"
VAL_SOURCE = r"C:\Users\ADMIN\Downloads\archive (7)\NEU-DET\validation\images"

# Destination folders
TRAIN_DEST = r"dataset\images\train"
VAL_DEST = r"dataset\images\validation"

# Create destination folders
os.makedirs(TRAIN_DEST, exist_ok=True)
os.makedirs(VAL_DEST, exist_ok=True)


def copy_images(source_folder, destination_folder):
    count = 0

    # Loop through each class folder
    for class_folder in os.listdir(source_folder):

        class_path = os.path.join(source_folder, class_folder)

        if not os.path.isdir(class_path):
            continue

        # Loop through each image
        for image in os.listdir(class_path):

            if image.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

                src = os.path.join(class_path, image)
                dst = os.path.join(destination_folder, image)

                shutil.copy2(src, dst)
                count += 1

    return count


print("Copying Training Images...")
train_count = copy_images(TRAIN_SOURCE, TRAIN_DEST)

print("Copying Validation Images...")
val_count = copy_images(VAL_SOURCE, VAL_DEST)

print("\nFinished!")
print(f"Training Images   : {train_count}")
print(f"Validation Images : {val_count}")