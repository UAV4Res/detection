import cv2
import numpy as np
import time
import math
import random
from ultralytics import YOLO

videoSource = "test.mp4"

cap = cv2.VideoCapture(videoSource)
net = YOLO("best_top.pt")
processFrame = True

cv2.namedWindow("Person Tracking", cv2.WINDOW_AUTOSIZE)
font = cv2.FONT_HERSHEY_PLAIN

while True:
    startTime = time.time()
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame[:, 250: -100], (800, 800))
    if processFrame == True:
        boxes = net.predict(frame)
        boxes = boxes[0].boxes.xywh

        curCoords = []
        for box in boxes:
            box[0] = box[0] - box[2] / 2
            box[1] = box[1] - box[3] / 2
            curCoords.append([int(box[0])+int(box[2]/2), int(box[1])+int(box[3]/2)])
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), font, 1, (0, 255, 0), 1)
        num_people = len(boxes)
        cv2.putText(frame, f'Count: {num_people}', (10, 30), font, 2, (0, 255, 0), 2)

    key = None
    key = cv2.waitKey(50) & 0xFF
    if key  == ord('q'):
        break

    cv2.imshow("Person Tracking", frame)
    
cap.release()
cv2.destroyAllWindows()
