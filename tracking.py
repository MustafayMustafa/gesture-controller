import time

import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


last_time = current_time = 0


while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # if landmark detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, land_mark in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(land_mark.x*w), int(land_mark.y*h)
                print(f"ID: {id}, x: {cx}, y: {cy}")

            mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # calculate, draw fps
    current_time = time.time()
    fps = 1/(current_time-last_time)
    last_time = current_time
    cv2.putText(
        img=image, 
        text=str(int(fps)), 
        org=(20, 70), 
        fontFace=cv2.FONT_HERSHEY_PLAIN, 
        fontScale=5, 
        color=(255, 255, 255), 
        thickness=1
    )

    cv2.imshow('Image', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()