import cv2
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO

try:
    model = YOLO("yolov8n.pt")
except Exception:
    model = YOLO("./best.pt")

GPIO.setmode(GPIO.BCM)
SWITCH_X = 17
SWITCH_Y = 27
GPIO.setup(SWITCH_X, GPIO.OUT)
GPIO.setup(SWITCH_Y, GPIO.OUT)

areas = {}
selected_area = None
object_detected = False
categories = [
    "electronics", "glass", "metal", "organic_waste",
    "paper_and_cardboard", "plastic", "textile", "wood"
]

def select_area(event, x, y, flags, param):
    global selected_area
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_area = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_area.append((x, y))

def configure_areas(cap):
    global areas
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area_name, coords in areas.items():
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, area_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Configuratie", frame)
        key = cv2.waitKey(1)

        if selected_area and len(selected_area) == 2:
            x1, y1 = selected_area[0]
            x2, y2 = selected_area[1]
            category = choose_category()
            areas[category] = [(x1, y1), (x2, y2)]
            selected_area = None

        if key == 27:
            break

def choose_category():
    for idx, category in enumerate(categories, 1):
        print(f"{idx}. {category}")
    while True:
        try:
            choice = int(input("Voer het nummer in van de categorie: "))
            if 1 <= choice <= len(categories):
                return categories[choice - 1]
        except ValueError:
            pass

def enable_x_axis():
    GPIO.output(SWITCH_X, GPIO.HIGH)

def disable_x_axis():
    GPIO.output(SWITCH_X, GPIO.LOW)

def enable_y_axis():
    GPIO.output(SWITCH_Y, GPIO.HIGH)

def disable_y_axis():
    GPIO.output(SWITCH_Y, GPIO.LOW)

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    return cap

def live_identification(cap):
    global object_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area_name, coords in areas.items():
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, area_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if "start" in areas and not object_detected:
            start_coords = areas["start"]
            x1, y1 = start_coords[0]
            x2, y2 = start_coords[1]
            cropped_frame = frame[y1:y2, x1:x2]
            results = model(cropped_frame)

            for r in results:
                for obj in r.boxes:
                    category = model.names[int(obj.cls)]
                    if category in areas:
                        target_coords = areas[category]
                        enable_x_axis()
                        time.sleep(2)
                        disable_x_axis()
                        object_detected = True

        cv2.imshow("Live Identificatie", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap = initialize_camera()
if cap:
    cv2.namedWindow("Configuratie")
    cv2.setMouseCallback("Configuratie", select_area)
    configure_areas(cap)
    cv2.destroyWindow("Configuratie")

    cv2.namedWindow("Live Identificatie")
    live_identification(cap)

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
