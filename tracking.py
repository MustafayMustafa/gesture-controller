import time
import cv2
import mediapipe as mp


class Detector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        DEBUG_MODE=False,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.DEBUG_MODE = DEBUG_MODE

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )

    def get_hands(self):
        if self.results.multi_hand_landmarks:
            return self.results.multi_hand_landmarks

        return []

    def find_hands(self, image, draw=False):
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for hand_landmarks in self.get_hands():
            if draw:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return image

    def get_landmark_position(self, image, landmark_index):
        hands = self.get_hands()
        landmarks = []
        for hand in hands:
            landmarks = [(id, land_mark) for id, land_mark in enumerate(hand.landmark)]
            landmark = landmarks[landmark_index][1]
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmarks.append([id, cx, cy])

            if self.DEBUG_MODE:
                print(f"ID: {id}, x: {cx}, y: {cy}")
                cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return landmarks


def main():
    cap = cv2.VideoCapture(0)
    last_time = current_time = 0
    debug = True
    detector = Detector(DEBUG_MODE=debug)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = detector.find_hands(image=image, draw=True)
        landmark_position_list = detector.get_landmark_position(image, 1)

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


if __name__ == "__main__":
    main()
