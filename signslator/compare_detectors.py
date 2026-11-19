"""
Same scene, both detectors: compare landmark jitter and detection cost.

Jitter = mean frame-to-frame change in landmark positions. Lower is steadier.
This is a proxy for landmark quality — noise in the input features directly
caps recognition accuracy.

NOTE: this runs on whatever is in front of the camera in this session. The
absolute numbers are scene-dependent; the *comparison* is what matters.
"""

import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    HolisticLandmarker, HolisticLandmarkerOptions,
    RunningMode,
)

HAND_MODEL = "signslator/models/hand_landmarker.task"
HOLISTIC_MODEL = "signslator/models/holistic_landmarker.task"


def run_hand(seconds=6):
    base = python.BaseOptions(model_asset_path=HAND_MODEL)
    opts = HandLandmarkerOptions(base_options=base, running_mode=RunningMode.VIDEO,
                                 num_hands=2, min_hand_detection_confidence=0.5)
    lm = HandLandmarker.create_from_options(opts)
    return _measure(lm, "hand", seconds, holistic=False)


def run_holistic(seconds=6):
    base = python.BaseOptions(model_asset_path=HOLISTIC_MODEL)
    opts = HolisticLandmarkerOptions(base_options=base, running_mode=RunningMode.VIDEO,
                                     min_hand_landmarks_confidence=0.5)
    lm = HolisticLandmarker.create_from_options(opts)
    return _measure(lm, "holistic", seconds, holistic=True)


def _measure(lm, name, seconds, holistic):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frames = 0
    detected = 0
    t_detect = 0.0
    jitters = []
    prev = None
    t0 = time.time()

    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        a = time.time()
        res = lm.detect_for_video(img, int(time.time() * 1000))
        t_detect += time.time() - a

        if holistic:
            hands = (res.left_hand_landmarks or []) + (res.right_hand_landmarks or [])
            if hands:
                pts = np.array([[lm_.x, lm_.y] for lm_ in hands], dtype=np.float64)
            else:
                pts = None
        else:
            hands = res.hand_landmarks
            if hands:
                pts = np.array([[lm_.x, lm_.y] for lm_ in hands[0]], dtype=np.float64)
            else:
                pts = None

        if pts is not None:
            detected += 1
            if prev is not None and pts.shape == prev.shape:
                jitters.append(float(np.abs(pts - prev).mean()))
            prev = pts
        else:
            prev = None

    cap.release()
    lm.close()

    j = np.array(jitters) if jitters else np.array([0.0])
    return {
        "name": name,
        "ms": t_detect / max(1, frames) * 1000,
        "fps": frames / seconds,
        "detected": 100 * detected / max(1, frames),
        "jitter": j.mean() * 1000,
        "p95": np.percentile(j, 95) * 1000,
        "n": len(jitters),
    }


def main():
    print("Measuring both detectors on the same scene...\n")
    hand = run_hand()
    time.sleep(1)
    hol = run_holistic()

    print("=" * 62)
    print(f"{'':16} {'HandLandmarker':>18} {'Holistic':>14}")
    print("-" * 62)
    print(f"{'detect cost':16} {hand['ms']:>15.1f} ms {hol['ms']:>11.1f} ms")
    print(f"{'frame rate':16} {hand['fps']:>15.1f}    {hol['fps']:>11.1f}")
    print(f"{'hand present':16} {hand['detected']:>14.0f}% {hol['detected']:>10.0f}%")
    print(f"{'jitter mean':16} {hand['jitter']:>15.2f} mm {hol['jitter']:>10.2f} mm")
    print(f"{'jitter p95':16} {hand['p95']:>15.2f} mm {hol['p95']:>10.2f} mm")
    print("=" * 62)

    if hand["jitter"] < hol["jitter"]:
        print(f"\nHandLandmarker is {hol['jitter']/hand['jitter']:.1f}x steadier.")
    else:
        print(f"\nHandLandmarker is NOT steadier on this scene "
              f"({hand['jitter']:.2f} vs {hol['jitter']:.2f} mm).")
    print("\nScene-dependent — re-run with a real hand in front of the camera.")


if __name__ == "__main__":
    main()
