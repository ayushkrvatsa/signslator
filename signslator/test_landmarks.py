"""
Quick headless test: capture one frame, run the hand landmarker, save output image.
Verifies the landmark pipeline works end-to-end without needing a GUI window.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import os
import time
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("FAIL: Could not capture frame")
    exit(1)

print(f"Frame captured: {frame.shape}")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = landmarker.detect(mp_image)

display = frame.copy()

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# Draw hands
hands = getattr(result, "hand_landmarks", None) or []
handed = getattr(result, "handedness", None) or []
print(f"Hands detected: {len(hands)}")

for k, hand_lms in enumerate(hands):
    label = handed[k][0].category_name if k < len(handed) and handed[k] else "?"
    print(f"  hand {k} ({label}): {len(hand_lms)} landmarks")
    for lm in hand_lms:
        x, y = int(lm.x * display.shape[1]), int(lm.y * display.shape[0])
        cv2.circle(display, (x, y), 4, (255, 0, 0), -1)
    for i, j in HAND_CONN:
        if i < len(hand_lms) and j < len(hand_lms):
            xi = int(hand_lms[i].x * display.shape[1])
            yi = int(hand_lms[i].y * display.shape[0])
            xj = int(hand_lms[j].x * display.shape[1])
            yj = int(hand_lms[j].y * display.shape[0])
            cv2.line(display, (xi, yi), (xj, yj), (255, 100, 0), 2)

if not hands:
    print("No hands detected")

os.makedirs("captures", exist_ok=True)
out_path = "captures/landmark_test.png"
cv2.imwrite(out_path, display)
print(f"\nSaved: {out_path}")
print("Headless test PASSED — landmark pipeline works.")

landmarker.close()
