from ultralytics import YOLO
from ultralytics.solutions import ai_gym
import cv2

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("pullups.mp4")
assert cap.isOpened(), "Error reading video file"

gym_object = ai_gym.AIGym()  # init AI GYM module
gym_object.set_args(line_thickness=2,
                    view_img=True,
                    pose_type="pullup",
                    kpts_to_check=[6, 8, 10])

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        exit(0)
    frame_count += 1
    results = model.predict(im0, verbose=False)
    im0 = gym_object.start_counting(im0, results, frame_count)