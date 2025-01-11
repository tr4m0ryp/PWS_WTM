import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

model = YOLO("yolov8n.pt")

GPIO.setmode(GPIO.BCM)
RELAY_X = 17  # Relay voor de motor op de x-as
RELAY_Y = 27  # Relay voor de motor op de y-as
GPIO.setup(RELAY_X, GPIO.OUT)
GPIO.setup(RELAY_Y, GPIO.OUT)

# Variabelen
areas = {}
selected_area = None
object_detected = False
tracker = None
highlight_color = (0, 255, 255)

categories = [
    "electronics", "glass", "metal", "organic_waste",
    "paper_and_cardboard", "plastic", "textile", "wood"
]

def enable_motor(relay):
    GPIO.output(relay, GPIO.HIGH)

def disable_motor(relay):
    GPIO.output(relay, GPIO.LOW)

def select_area(event, x, y, flags, param):
    global selected_area
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_area = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_area.append((x, y))

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

def configure_areas():
    global areas
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area_name, coords in areas.items():
            x1, y1, x2, y2 = *coords[0], *coords[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, area_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Configuratie", frame)
        key = cv2.waitKey(1)

        if selected_area and len(selected_area) == 2:
            x1, y1 = selected_area[0]
            x2, y2 = selected_area[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Configuratie", frame)
            cv2.waitKey(500)

            category = choose_category()
            areas[category] = [(x1, y1), (x2, y2)]
            selected_area = None

        if key == 27:
            break

def live_identification():
    global tracking_object, tracker, object_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area_name, coords in areas.items():
            x1, y1, x2, y2 = *coords[0], *coords[1]
            color = highlight_color if area_name == "highlight" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, area_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not object_detected and "start" in areas:
            start_coords = areas["start"]
            x1, y1, x2, y2 = *start_coords[0], *start_coords[1]
            cropped_frame = frame[y1:y2, x1:x2]
            results = model(cropped_frame)

            for r in results:
                for obj in r.boxes:
                    category = model.names[int(obj.cls)]
                    if category in areas:
                        target_coords = areas[category]
                        object_detected = True
                        areas["highlight"] = target_coords

                        bbox = obj.xyxy[0].tolist()
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, tuple(map(int, bbox)))
                      
        if tracker and object_detected:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.circle(frame, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 0, 255), 2)

                target_coords = areas.get("highlight")
                if target_coords:
                    tx1, ty1, tx2, ty2 = *target_coords[0], *target_coords[1]
                    if tx1 <= x + w // 2 <= tx2 and ty1 <= y + h // 2 <= ty2:
                        cv2.putText(frame, "Object gesorteerd!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Live Identificatie", frame)
                        cv2.waitKey(1000)

                        print("Object gesorteerd! Het systeem is klaar voor het volgende object.")  # Melding in de console
                        object_detected = False
                        areas.pop("highlight", None)
                        tracker = None

        cv2.imshow("Live Identificatie", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

cv2.namedWindow("Configuratie")
cv2.setMouseCallback("Configuratie", select_area)
configure_areas()
cv2.destroyWindow("Configuratie")

cv2.namedWindow("Live Identificatie")
live_identification()

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
