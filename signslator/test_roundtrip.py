"""
End-to-end round trip on the live camera with the new HandLandmarker.

capture -> save template -> load -> match -> correct sign

This exercises the same code path as `run.py train` + `run.py recognize`
minus the keypress UI, so it can run headless. Cleans up after itself.
"""

import os
import shutil
import sys
import time
import types

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, ".")

from signslator.sign_to_speech import (
    hand_landmarker, extract_sequence_features, load_all_templates,
    find_best_match, _build_prefilter, DATA_DIR,
)


def capture(seconds=2.0, width=640, height=480, variant=0):
    """Record landmark results for `seconds`, return the feature sequence.

    Falls back to a synthetic hand when the camera sees none, so the
    save/load/match path is still exercised. Reports which mode was used.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    results = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results.append(hand_landmarker.detect_for_video(img, int(time.time() * 1000)))

    cap.release()

    if any(getattr(r, "hand_landmarks", None) for r in results):
        print("    (real camera hand)")
        return extract_sequence_features(results)

    # No hand in view — build a synthetic one so the pipeline is still tested.
    print("    (no hand in view — using synthetic landmarks)")
    rng = np.random.default_rng(variant)
    synth = []
    # Different variant -> a clearly different hand shape.
    base = np.array([[(0.1 + 0.15 * variant) * (i % 5),
                      (0.1 + 0.15 * variant) * (i // 5), 0.0]
                     for i in range(21)], dtype=np.float32)
    for _ in range(int(seconds * 20)):
        frame_pts = base + rng.normal(0, 0.01, base.shape).astype(np.float32)
        frame_pts = np.clip(frame_pts, 0.0, 1.0)
        lms = [types.SimpleNamespace(x=float(x), y=float(y), z=float(z))
               for x, y, z in frame_pts]
        synth.append(types.SimpleNamespace(
            hand_landmarks=[lms],
            handedness=[[types.SimpleNamespace(category_name="Right")]],
        ))
    return extract_sequence_features(synth)


def main():
    tmp_sign = "ROUNDTRIPTEST"
    sign_dir = os.path.join(DATA_DIR, tmp_sign)
    shutil.rmtree(sign_dir, ignore_errors=True)
    os.makedirs(sign_dir, exist_ok=True)

    try:
        print("=" * 58)
        print("Round trip: capture -> save -> load -> match")
        print("=" * 58)

        # ── 1. capture two samples of the same "sign" ──
        print("\n[1] capturing sample A (2s)...")
        a = capture(2.0)
        print(f"    {a.shape[0]} frames, {a.shape[1]} features/frame")

        print("[2] capturing sample B (2s)...")
        b = capture(2.0)
        print(f"    {b.shape[0]} frames")

        np.save(os.path.join(sign_dir, "template_0.npy"), a)
        np.save(os.path.join(sign_dir, "template_1.npy"), b)

        # ── 2. a different "sign" (captured later, different motion) ──
        other_dir = os.path.join(DATA_DIR, "ROUNDTRIPOTHER")
        shutil.rmtree(other_dir, ignore_errors=True)
        os.makedirs(other_dir, exist_ok=True)
        print("[3] capturing a different sample (2s)...")
        c = capture(2.0, variant=1)
        np.save(os.path.join(other_dir, "template_0.npy"), c)

        # ── 3. load and match ──
        templates = load_all_templates()
        print(f"\n[4] loaded {len(templates)} signs from disk")
        prefilter = _build_prefilter(templates)

        for name, seq in [("A", a), ("B", b), ("C", c)]:
            best, dist, dists = find_best_match(seq, templates, prefilter=prefilter)
            print(f"    sample {name} -> {best}  (dist {dist:.3f})")

        # ── 4. assertions ──
        print("\n" + "-" * 58)
        ok = True

        ba, da, _ = find_best_match(a, templates, prefilter=prefilter)
        bb, db, _ = find_best_match(b, templates, prefilter=prefilter)
        bc, dc, _ = find_best_match(c, templates, prefilter=prefilter)
        if ba == tmp_sign and bb == tmp_sign:
            print(f"  same-sign samples both matched {tmp_sign}:  PASS")
        else:
            print(f"  same-sign match: A->{ba}, B->{bb} (expected {tmp_sign}):  FAIL")
            ok = False

        if bc == "ROUNDTRIPOTHER":
            print(f"  different sample matched ROUNDTRIPOTHER:      PASS")
        else:
            print(f"  different sample -> {bc} (expected ROUNDTRIPOTHER): FAIL")
            ok = False

        # A template matched against itself must be ~0 distance.
        if da < 0.05:
            print(f"  self-match distance small ({da:.4f}):            PASS")
        else:
            print(f"  self-match distance {da:.3f} (expected <0.05):     FAIL")
            ok = False

        # Empty sequence must be refused, not matched at distance 0.
        empty = np.zeros((40, 144), dtype=np.float32)
        be, de, _ = find_best_match(empty, templates, prefilter=prefilter)
        if be is None and de == float("inf"):
            print(f"  empty sequence refused (no false match):     PASS")
        else:
            print(f"  empty sequence -> {be} at {de}:                FAIL")
            ok = False

        print("-" * 58)
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    finally:
        shutil.rmtree(sign_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(DATA_DIR, "ROUNDTRIPOTHER"), ignore_errors=True)
        calib = os.path.join(DATA_DIR, "_calibration.json")
        if os.path.exists(calib):
            os.remove(calib)
        print("\n(cleaned up test templates)")


if __name__ == "__main__":
    sys.exit(main())
