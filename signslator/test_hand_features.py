"""
Verify the HandLandmarker pipeline: slot assignment, feature layout, and
live detection. Also measures landmark jitter (frame-to-frame stability) so the
"is detection better" claim is a number, not an opinion.
"""

import sys
import time
import types

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, ".")

from signslator.sign_to_speech import (
    _hand_slots, extract_hand_features, _draw_landmarks, hand_landmarker,
)


def make_result(hands, labels=None):
    """Build a fake HandLandmarkerResult for slot-assignment tests."""
    r = types.SimpleNamespace()
    r.hand_landmarks = []
    for pts in hands:
        lms = []
        for x, y, z in pts:
            lms.append(types.SimpleNamespace(x=float(x), y=float(y), z=float(z)))
        r.hand_landmarks.append(lms)
    r.handedness = []
    for i in range(len(hands)):
        lab = (labels[i] if labels and i < len(labels) else "Left")
        r.handedness.append([types.SimpleNamespace(category_name=lab)])
    return r


def grid(x0, y0):
    """A plausible 21-point hand shape anchored near (x0, y0)."""
    pts = []
    for i in range(21):
        pts.append((x0 + 0.01 * (i % 5), y0 + 0.01 * (i // 5), 0.0))
    return pts


def test_slots():
    print("=" * 58)
    print("1. Slot assignment")
    print("=" * 58)
    ok = 0
    total = 0

    # One hand labelled Right, sitting on the image's left.
    r = make_result([grid(0.2, 0.5)], ["Right"])
    left, right = _hand_slots(r)
    total += 1
    good = right is not None and left is None
    ok += good
    print(f"  single 'Right' hand -> right slot: {'PASS' if good else 'FAIL'}")

    # Two hands: Right on image-left, Left on image-right.
    r = make_result([grid(0.2, 0.5), grid(0.8, 0.5)], ["Right", "Left"])
    left, right = _hand_slots(r)
    total += 1
    good = (left is not None and right is not None
            and left[:, 0].mean() > right[:, 0].mean())
    ok += good
    print(f"  two hands, slots not crossed:       {'PASS' if good else 'FAIL'}")

    # Missing label -> x-position fallback.
    r = make_result([grid(0.8, 0.5), grid(0.2, 0.5)], [None, None])
    left, right = _hand_slots(r)
    total += 1
    good = (left is not None and right is not None
            and left[:, 0].mean() < right[:, 0].mean())
    ok += good
    print(f"  no labels -> x-position fallback:   {'PASS' if good else 'FAIL'}")

    # Empty result.
    r = make_result([])
    left, right = _hand_slots(r)
    total += 1
    good = left is None and right is None
    ok += good
    print(f"  no hands -> (None, None):           {'PASS' if good else 'FAIL'}")

    print(f"\n  {ok}/{total} passed\n")
    return ok == total


def test_features():
    print("=" * 58)
    print("2. Feature layout")
    print("=" * 58)
    r = make_result([grid(0.2, 0.5), grid(0.8, 0.5)], ["Right", "Left"])
    f = extract_hand_features(r)
    shape_ok = f.shape == (144,)
    nonzero = int(np.count_nonzero(f))
    left_nz = int(np.count_nonzero(f[0:63]))
    right_nz = int(np.count_nonzero(f[63:126]))
    deriv_nz = int(np.count_nonzero(f[126:144]))
    print(f"  shape (144,):            {'PASS' if shape_ok else 'FAIL'}")
    print(f"  non-zero values:         {nonzero}/144")
    print(f"  left block  [0:63]:      {left_nz} non-zero")
    print(f"  right block [63:126]:    {right_nz} non-zero")
    print(f"  derived    [126:144]:    {deriv_nz} non-zero")
    ok = shape_ok and left_nz > 0 and right_nz > 0 and deriv_nz > 0
    print(f"\n  {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_live(seconds=6):
    print("=" * 58)
    print("3. Live detection + jitter")
    print("=" * 58)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frames = 0
    with_hand = 0
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
        res = hand_landmarker.detect_for_video(img, int(time.time() * 1000))
        t_detect += time.time() - a

        hands = getattr(res, "hand_landmarks", None) or []
        if hands:
            with_hand += 1
            pts = np.array([[lm.x, lm.y] for lm in hands[0]], dtype=np.float64)
            if prev is not None and pts.shape == prev.shape:
                jitters.append(float(np.abs(pts - prev).mean()))
            prev = pts
        else:
            prev = None

    cap.release()

    print(f"  frames:        {frames} in {seconds}s = {frames/seconds:.1f} fps")
    print(f"  detection:     {t_detect/max(1,frames)*1000:.1f} ms/frame")
    print(f"  hand in view:  {with_hand}/{frames} ({100*with_hand/max(1,frames):.0f}%)")
    if jitters:
        j = np.array(jitters)
        print(f"  jitter:        mean {j.mean()*1000:.2f} mm/frame  "
              f"(p95 {np.percentile(j,95)*1000:.2f})  n={len(j)}")
        print(f"                  lower = steadier landmarks")
    else:
        print("  jitter:        no consecutive hand frames to measure")
    print()
    return True


if __name__ == "__main__":
    a = test_slots()
    b = test_features()
    test_live()
    print("=" * 58)
    print("RESULT:", "ALL PASS" if (a and b) else "FAILURES — see above")
    print("=" * 58)
