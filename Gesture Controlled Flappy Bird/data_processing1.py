import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

output_file = 'hand_landmarks.csv'

def extract_landmarks(image_path, label):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    print(f"Processing: {image_path}")
    print(f"Results: {results}")

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])
        landmarks.append(label)
        return landmarks
    return None


    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.hand_landmarks.landmark:
            landmarks.extend([hand_landmarks.x, hand_landmarks.y, hand_landmarks.z])
        landmarks.append(label)
        return landmarks
    return None

header = [f'point_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    folder = 'dataset_images'
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            label = label_folder
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                data = extract_landmarks(img_path, label)
                if data:
                    writer.writerow(data)

print(f'Landmarks saved to {output_file}')
