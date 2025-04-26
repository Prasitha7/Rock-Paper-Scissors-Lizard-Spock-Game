import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import preprocess_frame

# Load model
model = tf.keras.models.load_model('models/gesture_model.h5')
CATEGORIES = ['rock', 'paper', 'scissors', 'lizard', 'spock']
IMG_SIZE = 224

# Webcam
cap = cv2.VideoCapture(0)
window_name = "Live Gesture Prediction"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("[INFO] Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    original, skin_mask, largest_only = preprocess_frame(frame)

    # Prepare input for model
    resized = cv2.resize(largest_only, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    input_img = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    predictions = model.predict(input_img)[0]
    prediction_confidences = list(zip(CATEGORIES, predictions))
    prediction_confidences.sort(key=lambda x: x[1], reverse=True)

    # Overlay prediction results
    display_frame = original.copy()
    y_offset = 30

    for idx, (label, confidence) in enumerate(prediction_confidences):
        text = f"{label}: {confidence * 100:.2f}%"
        if idx == 0:
            # Highlight top-1 prediction
            cv2.putText(display_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_offset += 30

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
