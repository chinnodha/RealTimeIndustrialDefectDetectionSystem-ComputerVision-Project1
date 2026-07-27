import cv2

for index in range(4):
    print(f"\nTrying camera {index}...")

    cap = cv2.VideoCapture(index)

    if cap.isOpened():
        print(f"Camera {index} opened successfully!")

        ret, frame = cap.read()
        print("Frame:", ret)

        if ret:
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

        cap.release()
    else:
        print(f"Camera {index} could not be opened.")