import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1080, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]
        cv2.rectangle(img, (0, 325), (220, 600), (255, 255, 200), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (65, 520), cv2.FONT_HERSHEY_PLAIN,
                    10, (100, 100, 100), 25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (870, 70), cv2.FONT_HERSHEY_PLAIN,
                6, (50, 0, 255), 3)
    cv2.imshow("Image", img)
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break