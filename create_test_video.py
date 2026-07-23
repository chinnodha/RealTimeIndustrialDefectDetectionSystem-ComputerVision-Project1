import cv2
import os
from glob import glob

# Folder containing validation images
image_folder = "dataset/images/validation"

# Output video name
output_video = "test_video.mp4"

# Frames per second
fps = 5

# Get all images
image_files = sorted(glob(os.path.join(image_folder, "*.jpg")))[:20]

# If no JPG images, try PNG
if len(image_files) == 0:
    image_files = sorted(glob(os.path.join(image_folder, "*.png")))[:20]

if len(image_files) == 0:
    print("No images found!")
    exit()

# Read first image to get size
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Creating video...")

for img_path in image_files:
    frame = cv2.imread(img_path)

    # Resize if needed
    frame = cv2.resize(frame, (width, height))

    # Display filename on frame (optional)
    filename = os.path.basename(img_path)
    cv2.putText(
        frame,
        filename,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Add each image multiple times to slow down playback
    for _ in range(10):
        video.write(frame)

video.release()

print(f"\nVideo saved as: {output_video}")