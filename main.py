import cv2
import mediapipe as mp
import HandTrackingModule as htm

def openFingersTest(img, detector):
    openlist = detector.openFingers(img)
    if (len(openlist) > 5):
        print('Right: ')
        print(openlist[:5])
        print('Left: ')
        print(openlist[5:])
    else:
        print(openlist)


capture = cv2.VideoCapture(0)


detector = htm.Hand()


while True:
    success, img = capture.read()
    img = detector.findHands(img)

    #openFingersTest(img, detector)
    #print(detector.isFist(img))
    #print(detector.isThumbsUp(img))
    print(detector.isThumbsDown(img))

    img = cv2.flip(img, 1)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)
