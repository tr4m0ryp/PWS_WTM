import cv2
import time
import math
from ultralytics import YOLO


model_path = 'C:/Users/Moussa/Pictures/Screenshots/best.pt'
model = YOLO(model_path)
print(model.names)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Kan de camera niet openen.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

DETECTION_INTERVAL = 5.0
last_detection_time = 0.0

last_closest_box = None
last_label = None

while True:
    ret, frame = camera.read()
    if not ret:
        print("Kan geen frame lezen van de camera.")
        break

    current_time = time.time()

    if current_time - last_detection_time >= DETECTION_INTERVAL:
        last_detection_time = current_time

        results = model.predict(frame, verbose=False)
        
        boxes = results[0].boxes if len(results) > 0 else []

        frame_height, frame_width = frame.shape[:2]
        center_frame_x = frame_width // 2
        center_frame_y = frame_height // 2

        closest_box = None
        closest_distance = float('inf')

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            bx_center = (x1 + x2) / 2
            by_center = (y1 + y2) / 2

            dist = math.sqrt((bx_center - center_frame_x)**2 + (by_center - center_frame_y)**2)

            if dist < closest_distance:
                closest_distance = dist
                closest_box = box

        if closest_box is not None:
            x1, y1, x2, y2 = map(int, closest_box.xyxy[0])
            confidence = closest_box.conf[0]
            class_id = int(closest_box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"

            last_closest_box = (x1, y1, x2, y2)
            last_label = label
        else:
            last_closest_box = None
            last_label = None
            
    if last_closest_box is not None:
        x1, y1, x2, y2 = last_closest_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Live Single Object (Center) Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
