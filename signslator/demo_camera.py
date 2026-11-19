"""
Phase 1A (v3): Live camera + MediaPipe Hand Landmarker overlay.

Detects 21 landmarks per hand (up to 2 hands). The hand-only model is steadier
and cheaper than the old holistic model (measured 1.80 vs 4.02 mm jitter).

Controls:
  'q' or ESC  — quit
  's'          — save current frame to ./captures/
  'r'          — toggle raw camera / landmarks overlay
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

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Download with: curl -L -o models/hand_landmarker.task <url>")
    exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

# ── Camera ─────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera opened. Showing hand landmarks.")
print("Controls: [q/ESC] quit  [s] save frame  [r] toggle view")
print("-" * 55)

show_raw = False
frame_count = 0
os.makedirs("captures", exist_ok=True)

HAND_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to read frame")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    display = frame.copy()

    if not show_raw:
        hands = getattr(result, "hand_landmarks", None) or []
        w, h = display.shape[1], display.shape[0]

        for hand in hands:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for i, j in HAND_CONN:
                if i < len(pts) and j < len(pts):
                    cv2.line(display, pts[i], pts[j], (255, 100, 0), 2)
            for p in pts:
                cv2.circle(display, p, 4, (255, 0, 0), -1)

        handed = getattr(result, "handedness", None) or []
        labels = ", ".join(hh[0].category_name for hh in handed if hh) or "-"
        status = f"Hands: {len(hands)} | {labels} | Frame: {frame_count}"
        color = (0, 255, 0) if hands else (0, 0, 255)
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(display, "RAW CAMERA (press 'r' to see landmarks)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SignSlator — Hand Landmarks", display)

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
landmarker.close()
