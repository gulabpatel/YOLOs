import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n-seg.pt")

# Open the video file
video_path = "cars.Mov"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened:
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Yolov8 segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and the display window
cap.release()
cv2.destroyAllWindows()