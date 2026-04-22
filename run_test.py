# test_camera.py — run this first to check your camera + model
import cv2
from ultralytics import YOLO

model = YOLO("my_model.pt")
cap   = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results    = model(frame, verbose=True)  # verbose=True prints detections
    annotated  = results[0].plot()           # draws all boxes automatically
    cv2.imshow("Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()