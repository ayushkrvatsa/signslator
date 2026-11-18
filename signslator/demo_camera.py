"""
Phase 1A (v2): Live camera + MediaPipe Holistic Landmarker overlay (MediaPipe 1.0 Tasks API).
Detects: 33 pose + 478 face + 21 left hand + 21 right hand = 553 landmarks total.

Controls:
  'q' or ESC  — quit
  's'          — save current frame + landmarks to ./captures/
  'r'          — toggle raw camera / landmarks overlay
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

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "holistic_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Download with: curl -L -o models/holistic_landmarker.task <url>")
    exit(1)

# ── Holistic Landmarker setup ─────────────────────────────────────────────────

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)
holistic = HolisticLandmarker.create_from_options(options)

# ── Camera ─────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera opened. Showing holistic landmarks (pose + face + hands).")
print("Controls: [q/ESC] quit  [s] save frame  [r] toggle view")
print("-" * 55)

show_raw = False
frame_count = 0
os.makedirs("captures", exist_ok=True)

# Hand landmark connections (21 points)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Pose landmark connections (33 points) — key skeletal chain
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 14), (0, 15),
    (14, 16), (15, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (14, 22), (15, 23),
    (22, 24), (23, 25),
    (24, 26), (25, 27),
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to read frame")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = holistic.detect_for_video(mp_image, int(time.time() * 1000))

    display = frame.copy()

    if not show_raw and result:
        # ── Hands only ──
        all_hands = (result.left_hand_landmarks or []) + (result.right_hand_landmarks or [])
        for lm in all_hands:
            x, y = int(lm.x * display.shape[1]), int(lm.y * display.shape[0])
            cv2.circle(display, (x, y), 4, (255, 0, 0), -1)

        # Draw hand connections
        left_count = len(result.left_hand_landmarks or [])
        for i, j in HAND_CONNECTIONS:
            if i < left_count and j < left_count:
                li, lj = all_hands[i], all_hands[j]
                cv2.line(display, (int(li.x*display.shape[1]), int(li.y*display.shape[0])),
                         (int(lj.x*display.shape[1]), int(lj.y*display.shape[0])),
                         (255, 100, 0), 2)
            elif left_count > 0 and (i + left_count) < len(all_hands) and (j + left_count) < len(all_hands):
                ri, rj = all_hands[i + left_count], all_hands[j + left_count]
                cv2.line(display, (int(ri.x*display.shape[1]), int(ri.y*display.shape[0])),
                         (int(rj.x*display.shape[1]), int(rj.y*display.shape[0])),
                         (255, 100, 0), 2)

        # ── Status ──
        l_hand = result.left_hand_landmarks is not None and len(result.left_hand_landmarks) > 0
        r_hand = result.right_hand_landmarks is not None and len(result.right_hand_landmarks) > 0
        face_ok = result.face_landmarks is not None
        status = f"Face: {'OK' if face_ok else '-'} | L: {'Y' if l_hand else '-'} | R: {'Y' if r_hand else '-'} | Frame: {frame_count}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "RAW CAMERA (press 'r' to see landmarks)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SignSlator — Holistic Landmarks", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print(f"Captured {frame_count} frames. Exiting.")
        break
    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"captures/frame_{ts}.png", display)
        print(f"Saved: captures/frame_{ts}.png")
    elif key == ord('r'):
        show_raw = not show_raw
        mode = "RAW" if show_raw else "LANDMARKS"
        print(f"Switched to: {mode}")

cap.release()
cv2.destroyAllWindows()
holistic.close()
