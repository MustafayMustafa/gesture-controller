import math
import time

import cv2
import numpy as np
import tracking
import osascript


cap = cv2.VideoCapture(0)
last_time = current_time = 0
debug = True
detector = tracking.Detector(min_detection_confidence=0.7, DEBUG_MODE=debug)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image = detector.find_hands(image=image, draw=True)
    x = detector.get_landmark_position(image, 4)
    y = detector.get_landmark_position(image, 8)

    if x and y:
        # measure distance between landmark 4, 8
        cv2.line(image, x, y, (255, 0, 255), 3)
        cx, cy = (x[0] + y[0]) // 2, (x[1] + y[1]) // 2
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        length = math.hypot(y[0] - x[0], y[1] - x[1])

        # length 30 - 300
        # volume 0 - 100
        volume = np.interp(length, [30, 300], [0, 100])
        print(length, volume)
        osascript.osascript(f"set volume output volume {int(volume)}")

        # if length < 50:
        # cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # might want to do in a separate thread

    # calculate, draw fps
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time
    cv2.putText(
        img=image,
        text=str(int(fps)),
        org=(20, 70),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=5,
        color=(255, 255, 255),
        thickness=1,
    )

    cv2.imshow("Image", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
