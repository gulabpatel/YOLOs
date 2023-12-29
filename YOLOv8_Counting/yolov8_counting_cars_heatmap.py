from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model

cap = cv2.VideoCapture("heatmap_cars1.mp4")  # Video file Path, webcam 0
assert cap.isOpened(), "Error reading video file"

# Region for object counting
count_reg_pts = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Heatmap Init
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_JET,
                     imw=cap.get(4),  # should same as im0 width
                     imh=cap.get(3),  # should same as im0 height
                     view_img=True,
                     count_reg_pts=count_reg_pts)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        exit(0)
    results = model.track(im0, persist=True)
    im0 = heatmap_obj.generate_heatmap(im0, tracks=results)