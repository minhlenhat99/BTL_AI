import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                # print(id, cx, cy)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break

cap.release()
cv2.destroyAllWindows()