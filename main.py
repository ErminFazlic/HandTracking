import cv2
import mediapipe as mp
import HandTrackingModule as htm

capture = cv2.VideoCapture(0)


detector = htm.Hand()


while True:
    success, img = capture.read()
    img = detector.findHands(img)

    print(detector.openFingers(img))



    img = cv2.flip(img, 1)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)
