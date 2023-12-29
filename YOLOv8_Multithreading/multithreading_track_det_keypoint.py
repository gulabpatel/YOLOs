import threading
import cv2
from ultralytics import YOLO


def run_tracker_in_thread(filename, model, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """
    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')
model3 = YOLO('yolov8n-pose.pt')
model4 = YOLO('yolov8n-seg.pt')
model5 = YOLO('yolov8n-seg.pt')

# Define the video files for the trackers
video_file1 = "carsdriving.mp4"  # Path to video file, 0 for webcam
video_file2 = "Helmet_construction.mp4"  # Path to video file, 0 for webcam, 1 for external camera
video_file3 = "tiger_walking_2.mp4" 
video_file4 = "menswalking1.mp4" 
video_file5 = 0

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)
tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(video_file3, model3, 3), daemon=True)
tracker_thread4 = threading.Thread(target=run_tracker_in_thread, args=(video_file4, model4, 4), daemon=True)
tracker_thread5 = threading.Thread(target=run_tracker_in_thread, args=(video_file5, model5, 5), daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()
tracker_thread3.start()
tracker_thread4.start()
tracker_thread5.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()
tracker_thread3.join()
tracker_thread4.join()
tracker_thread5.join()

# Clean up and close windows
cv2.destroyAllWindows()
