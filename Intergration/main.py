import cv2
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import os

model = YOLO("yolov8n.pt") #als dit niet werkt, wat weleens voorkomt delete de import from ultrautics import YOLO en plaats gewoon ./best.pt als file path. file zit in dezelfde map


GPIO.setmode(GPIO.BCM)
SWITCH_X = 17  # Schakelaar voor x-as
SWITCH_Y = 27  # Schakelaar voor y-as
GPIO.setup(SWITCH_X, GPIO.OUT)
GPIO.setup(SWITCH_Y, GPIO.OUT)
x_motor = RpiMotorLib.A4988Nema(5, 6, (21, 21, 21), "DRV8825")
y_motor = RpiMotorLib.A4988Nema(13, 19, (21, 21, 21), "DRV8825")

areas = {}
selected_area = None
tracking_object = False
tracker = None
object_detected = False
highlight_color = (0, 255, 255) 

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

def choose_category():
    print("\nSelecteer een categorie voor dit gebied:")
    for idx, category in enumerate(categories, 1):
        print(f"{idx}. {category}")
    while True:
        try:
            choice = int(input("Voer het nummer in van de categorie: "))
            if 1 <= choice <= len(categories):
                return categories[choice - 1]
            else:
                print("Ongeldige keuze. Kies een nummer uit de lijst.")
        except ValueError:
            print("Ongeldige invoer. Voer een nummer in.")

def configure_areas():
    print("Start configuratie:")
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
            print(f"Gebied '{category}' opgeslagen.")
            selected_area = None

        if key == 27:  # ESC knop, ctrl c werkt ook gwoon altijd
            print("Configuratie voltooid.")
            break

def move_to_area(target_coords):
    x_steps = target_coords[0][0]
    y_steps = target_coords[0][1]
    enable_x_axis()
    x_motor.motor_go(True, "Full", abs(x_steps), 0.005, False, 0.05)
    disable_x_axis()

    enable_y_axis()
    y_motor.motor_go(True, "Full", abs(y_steps), 0.005, False, 0.05)
    disable_y_axis()

# aansturen van motor, het idee is om switches gewoon high/low te doen soort motor aan/uit
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
        print("Fout bij openen van de camera!")
        return None
    print("Camera succesvol gestart.")
    return cap

def live_identification():
    global tracking_object, tracker, object_detected
    print("Plaats een object in het STARTgebied.")
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
                        print(f"Beweeg naar '{category}'.")
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
                        print("Object in bak gevallen.")
                        cv2.putText(frame, "Object gesorteerd! Plaats het volgende object.", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Live Identificatie", frame)
                        cv2.waitKey(1000)

                        object_detected = False
                        areas.pop("highlight", None)
                        tracker = None

        cv2.imshow("Live Identificatie", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap = initialize_camera()
if cap is None:
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
