import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


model = YOLO("yolov8n-seg.pt")
names = model.model.names
cap = cv2.VideoCapture("carsdriving.mp4")

out = cv2.VideoWriter('instance-segmentation.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0)
    clss = results[0].boxes.cls.cpu().tolist()
    masks = results[0].masks.xy

    annotator = Annotator(im0, line_width=2)

    for mask, cls in zip(masks, clss):
        annotator.seg_bbox(mask=mask,
                           mask_color=colors(int(cls), True),
                           det_label=names[int(cls)])

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()