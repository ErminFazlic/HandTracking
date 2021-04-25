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
    print(detector.countFingers(img))
    if(detector.isFist(img)):
        print('Fist!')
    if(detector.isThumbsUp(img)):
        print('Thumbs Up!')
    if(detector.isThumbsDown(img)):
        print('Thumbs Down!')
    if(detector.isOkSign(img)):
        print('OK!')



    img = cv2.flip(img, 1)
    cv2.imshow("Capture", img)
    cv2.waitKey(1)
