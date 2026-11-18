"""
Quick headless test: capture one frame, run holistic landmarker, save output image.
Verifies the landmark pipeline works end-to-end without needing a GUI window.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)
import os
import time
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "holistic_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)
holistic = HolisticLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("FAIL: Could not capture frame")
    exit(1)

print(f"Frame captured: {frame.shape}")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = holistic.detect(mp_image)

display = frame.copy()

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
POSE_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,14),(0,15),
    (14,16),(15,17),
    (16,18),(17,19),
    (18,20),(19,21),
    (14,22),(15,23),
    (22,24),(23,25),
    (24,26),(25,27),
]

# Draw pose
if result.pose_landmarks:
    print(f"Pose: {len(result.pose_landmarks)} landmarks")
    for lm in result.pose_landmarks:
        x, y = int(lm.x * display.shape[1]), int(lm.y * display.shape[0])
        cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
    for i, j in POSE_CONN:
        if i < len(result.pose_landmarks) and j < len(result.pose_landmarks):
            xi = int(result.pose_landmarks[i].x * display.shape[1])
            yi = int(result.pose_landmarks[i].y * display.shape[0])
            xj = int(result.pose_landmarks[j].x * display.shape[1])
            yj = int(result.pose_landmarks[j].y * display.shape[0])
            cv2.line(display, (xi, yi), (xj, yj), (0, 200, 0), 2)
else:
    print("No pose detected")

# Draw hands
left_ok = result.left_hand_landmarks is not None
right_ok = result.right_hand_landmarks is not None
print(f"Left hand: {'detected' if left_ok else 'not detected'}")
print(f"Right hand: {'detected' if right_ok else 'not detected'}")

for hand_lms in (result.left_hand_landmarks or []) + (result.right_hand_landmarks or []):
    hand_label = "LEFT" if hand_lms is result.left_hand_landmarks else "RIGHT"
    print(f"  {hand_label}: {len(hand_lms)} landmarks")
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

os.makedirs("captures", exist_ok=True)
out_path = "captures/landmark_test.png"
cv2.imwrite(out_path, display)
print(f"\nSaved: {out_path}")
print("Headless test PASSED — landmark pipeline works.")

holistic.close()
