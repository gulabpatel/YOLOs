from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("cars.mp4")
assert cap.isOpened(), "Error reading video file"

counter = object_counter.ObjectCounter()  # Init Object Counter
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        exit(0)
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)