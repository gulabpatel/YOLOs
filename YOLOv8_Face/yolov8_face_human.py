import cv2
import cvzone

from ultralytics import YOLO

video = "video.mp4"

cap = cv2.VideoCapture(0)
face_model = YOLO('yolov8n-face.pt') #https://github.com/akanametov/yolov8-face

while True:
    rt, video = cap.read()
    video = cv2.resize(video, (1020, 720))
    face_result = face_model.predict(video, conf=0.40)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            cvzone.cornerRect(video, [x1, y1, w, h], l=9, rt=3)

    cv2.imshow('frame', video)
    cv2.waitKey(1)