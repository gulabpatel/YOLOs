import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("shopping_mall2.mp4")

out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(cap.get(3)), int(cap.get(4))))

center_point = (-10, int(cap.get(4)))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotator = Annotator(im0, line_width=2)

    for box, track_id in zip(boxes, track_ids):
        annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
        annotator.visioneye(box, center_point)

    out.write(im0)
    cv2.imshow("visioneye-pinpoint__", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()