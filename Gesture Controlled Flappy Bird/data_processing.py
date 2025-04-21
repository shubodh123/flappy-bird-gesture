import cv2
import os

GESTURES = {
    "start": "Press 'o' to capture START gesture (Palm Open)",
    "up": "Press 'u' to capture UP gesture (Thumbs Up)",
    "down": "Press 'd' to capture DOWN gesture (Thumbs Down)",
    "still": "Press 't' to capture STILL gesture (Fist)",
}

DATASET_PATH = "dataset_images"
os.makedirs(DATASET_PATH, exist_ok=True)

for gesture in GESTURES.keys():
    os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Press 'q' to quit at any time.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, "Press key for gesture capture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for i, (key, msg) in enumerate(GESTURES.items()):
        cv2.putText(frame, msg, (50, 100 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Capture Hand Gestures", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    for gesture, msg in GESTURES.items():
        if key == ord(msg.split("'")[1]):
            img_path = os.path.join(DATASET_PATH, gesture, f"{len(os.listdir(os.path.join(DATASET_PATH, gesture)))}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved {gesture} image: {img_path}")

cap.release()
cv2.destroyAllWindows()
