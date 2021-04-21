import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, landmarks, mpHands.HAND_CONNECTIONS)

    img = cv2.flip(img, 1)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)
