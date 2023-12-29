from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8m-pose.pt')
#yolov8n-pose, yolov8s-pose, yolov8l-pose,yolov8x-pose,yolov8x-pose-p6

# RUn inference on the source
# results = model(source="gymnaste.mp4", show=True, conf=0.3, save=True)
results = model(source=0, show=True, conf=0.3, save=True)