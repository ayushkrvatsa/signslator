"""
Probe the MediaPipe HandLandmarker before rewiring the engine.

Checks: API surface, VIDEO-mode result shape, handedness vs. real position,
detection rate, and per-frame cost at two resolutions.
"""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

MODEL = "signslator/models/hand_landmarker.task"


def main():
    print("=" * 60)
    print("HandLandmarker probe")
    print("=" * 60)

    base = python.BaseOptions(model_asset_path=MODEL)
    opts = HandLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(opts)
    print("created OK")
    print("result fields:", [f for f in dir(landmarker) if not f.startswith("_")][:8])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frames = 0
    detected = 0
    both = 0
    t_detect = 0.0
    t0 = time.time()
    samples = []

    while time.time() - t0 < 8:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        a = time.time()
        res = landmarker.detect_for_video(img, int(time.time() * 1000))
        t_detect += time.time() - a

        hands = res.hand_landmarks
        if hands:
            detected += 1
            if len(hands) >= 2:
                both += 1
            if len(samples) < 8:
                info = []
                for i, h in enumerate(hands):
                    hs = res.handedness[i]
                    name = hs[0].category_name if hs else "?"
                    xs = [lm.x for lm in h]
                    info.append(f"{name}(x={sum(xs)/len(xs):.2f})")
                samples.append(f"{len(hands)} hands: " + ", ".join(info))

    cap.release()
    landmarker.close()

    el = time.time() - t0
    print(f"\nframes: {frames} in {el:.1f}s = {frames/el:.1f} fps")
    print(f"detection: {t_detect/frames*1000:.1f} ms/frame")
    print(f"hand present in {detected}/{frames} frames ({100*detected/max(1,frames):.0f}%)")
    print(f"both hands in  {both}/{frames} frames ({100*both/max(1,frames):.0f}%)")

    print("\n--- samples (handedness vs mean x) ---")
    print("  x~0.0 = left edge of image, x~1.0 = right edge")
    for s in samples:
        print("  " + s)

    if samples:
        print("\nNOTE: if a hand on the image's LEFT is labelled 'Right',")
        print("      MediaPipe handedness is mirrored for selfie view.")


if __name__ == "__main__":
    main()
