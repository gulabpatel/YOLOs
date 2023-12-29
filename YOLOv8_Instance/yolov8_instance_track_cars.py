import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
track_history = defaultdict(lambda: [])

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture("carsdriving.mp4")

out = cv2.VideoWriter('instance-segmentation-object-tracking.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.track(im0, persist=True)
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotator = Annotator(im0, line_width=2)

    for mask, track_id in zip(masks, track_ids):
        annotator.seg_bbox(mask=mask,
                           mask_color=colors(track_id, True),
                           track_label=str(track_id))

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()