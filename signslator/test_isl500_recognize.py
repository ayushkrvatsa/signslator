"""
Test: Use ISL500 templates with our DTW recognizer.

This verifies that pre-recorded ISL500 landmark data from other signers
can be used for recognition — no need to record yourself.

Run this on your machine with hands visible in front of the camera.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HolisticLandmarker, HolisticLandmarkerOptions, RunningMode,
)
import numpy as np
import time

from signslator.sign_to_speech import (
    extract_hand_features,
    extract_sequence_features,
    dtw_distance,
    find_best_match,
    _draw_landmarks,
    load_all_templates,
)

# ── Setup ──────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "holistic_landmarker.task")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)
holistic = HolisticLandmarker.create_from_options(options)

# ── Load ISL500 templates ──────────────────────────────────────────────────────

print("Loading ISL500 templates from data/signs/...")
templates = load_all_templates()
if not templates:
    print("No ISL500 templates found. Run: python isl500_loader.py --words Hello Yes No")
    exit(1)

print(f"Loaded {len(templates)} ISL500 signs:")
for sign, tlist in sorted(templates.items()):
    print(f"  {sign}: {len(tlist)} templates")
print()

# ── Live recognition against ISL500 ────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Showing ISL500 recognition. Do a sign in front of the camera.")
print("Press 'q' to quit.\n")

buffer_size = 60  # 2 seconds at 30fps
frame_buffer = []
result_buffer = []
last_match = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = holistic.detect_for_video(mp_image, int(time.time() * 1000))

    frame_buffer.append(frame)
    result_buffer.append(result)

    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)
        result_buffer.pop(0)

    display = frame.copy()

    if len(result_buffer) >= 15:
        live_seq = extract_sequence_features(result_buffer)
        best_sign, best_dist, all_dists = find_best_match(live_seq, templates)

        # Show all distances
        y = 50
        for sign_name, dist in sorted(all_dists.items(), key=lambda x: x[1])[:8]:
            color = (0, 255, 0) if sign_name == best_sign else (80, 80, 80)
            cv2.putText(display, f"{sign_name}: {dist:.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 16

        if best_sign and best_dist < 8.0:
            confidence = max(0, 1.0 - best_dist / 8.0)
            if confidence >= 0.3:
                label = f"→ ISL500: {best_sign} ({confidence:.0%})"
                color = (0, 255, 0) if best_sign != last_match else (0, 200, 255)
                cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, color, 2)
                if best_sign != last_match:
                    print(f"Matched: {best_sign} (dist={best_dist:.2f}, conf={confidence:.0%})")
                    last_match = best_sign
            else:
                cv2.putText(display, f"→ {best_sign} (low conf)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display, "→ No confident ISL500 match", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    _draw_landmarks(display, result)
    cv2.imshow("ISL500 Recognition Test", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("\nDone.")
