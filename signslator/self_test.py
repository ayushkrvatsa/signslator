"""
Measure recognition accuracy on YOUR hands.

Records a few samples of each sign you perform, then reports:
  - accuracy against YOUR OWN samples (should be high)
  - accuracy against ISL500 templates (the cross-signer gap)
  - a confusion matrix showing exactly which signs get mixed up

This turns "accuracy is slightly off" into a number and a specific fix.

Usage:
  python run.py self-test                 # interactive: pick signs, record, report
  python run.py self-test --sign HELLO    # just one sign
  python run.py self-test --samples 3     # samples per sign (default 3)
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HolisticLandmarker, HolisticLandmarkerOptions, RunningMode,
)

from signslator import get_vocabulary
from signslator.sign_to_speech import (
    extract_sequence_features, find_best_match, dtw_distance,
    _draw_landmarks, DATA_DIR, MODEL_PATH,
)

SELF_DIR = os.path.join(DATA_DIR, "_selftest")


def record_samples(sign: str, num_samples: int, duration: float, holistic) -> list:
    """Record N samples of a sign from the webcam. Returns feature sequences."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: camera unavailable")
        return []

    sequences = []
    for i in range(num_samples):
        print(f"\n  Sample {i+1}/{num_samples} for '{sign}'")
        print("  Get ready, then press SPACE to record")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"SPACE = record '{sign}' ({i+1}/{num_samples})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "q = cancel", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 180), 1)
            cv2.imshow("SignSlator Self-Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            if key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                return sequences

        print("  Recording... perform the sign now")
        results = []
        start = time.time()
        frames = 0
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results.append(holistic.detect_for_video(mp_img, int(time.time() * 1000)))

            display = frame.copy()
            progress = (time.time() - start) / duration
            cv2.rectangle(display, (0, 0), (int(640 * progress), 10), (0, 255, 0), -1)
            cv2.putText(display, f"recording {frames} frames", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("SignSlator Self-Test", display)
            cv2.waitKey(1)

        seq = extract_sequence_features(results)
        if seq.shape[0] >= 5:
            sequences.append(seq)
            print(f"  captured {seq.shape[0]} frames")
        else:
            print(f"  WARNING: only {seq.shape[0]} frames, skipped")

        time.sleep(0.4)

    cap.release()
    cv2.destroyAllWindows()
    return sequences


def evaluate(sequences_by_sign: dict, isl_templates: dict):
    """Compare user samples against (a) themselves and (b) ISL500."""
    print("\n" + "=" * 68)
    print("SELF-TEST RESULTS")
    print("=" * 68)

    # ── A. Within your own samples ──
    print("\n[A] You vs YOUR OWN samples (leave-one-out)")
    own_pool = {s: list(v) for s, v in sequences_by_sign.items()}
    c = n = 0
    for s, seqs in sequences_by_sign.items():
        for i in range(len(seqs)):
            pool = {}
            for k, v in own_pool.items():
                if k == s:
                    rem = [x for j, x in enumerate(v) if j != i]
                    if rem:
                        pool[k] = rem
                else:
                    pool[k] = v
            pred, d, _ = find_best_match(seqs[i], pool)
            ok = pred == s
            c += ok
            n += 1
    print(f"    {c}/{n} ({100*c/n:.0f}%)" if n else "    (not enough samples)")
    print("    → high here means the classifier works; the gap is signer-specific")

    # ── B. You vs ISL500 ──
    print("\n[B] You vs ISL500 templates (what live recognition actually uses)")
    all_templates = dict(isl_templates)
    for s, seqs in sequences_by_sign.items():
        all_templates.setdefault(s, []).extend(seqs)

    rows = []
    for s, seqs in sequences_by_sign.items():
        for i, seq in enumerate(seqs):
            # pool = ISL500 only (no user samples) — the honest cross-signer test
            pred, d, _ = find_best_match(seq, isl_templates)
            rows.append((s, pred, d))

    if rows:
        c = sum(1 for t, p, _ in rows if p == t)
        print(f"    {c}/{len(rows)} ({100*c/len(rows):.0f}%)")
        print()
        print(f"    {'YOU SIGNED':<12}{'RECOGNIZED AS':<14}{'DIST':>8}")
        print("    " + "-" * 34)
        for true_s, pred_s, dist in rows:
            mark = "ok  " if pred_s == true_s else "MISS"
            print(f"    {true_s:<12}{str(pred_s):<14}{dist:>8.3f}  {mark}")

        # confusion summary
        conf = {}
        for t, p, _ in rows:
            if t != p:
                conf[(t, p)] = conf.get((t, p), 0) + 1
        if conf:
            print("\n    Confusions:")
            for (t, p), k in sorted(conf.items(), key=lambda x: -x[1]):
                print(f"      {t} → {p}  ({k}×)")

    # ── C. You vs hybrid (ISL500 + your samples) ──
    print("\n[C] You vs HYBRID pool (ISL500 + your recordings)")
    c = n = 0
    for s, seqs in sequences_by_sign.items():
        for i in range(len(seqs)):
            pool = {k: list(v) for k, v in all_templates.items()}
            # remove this sample from the pool
            own = [x for j, x in enumerate(pool.get(s, []))
                   if j < len(isl_templates.get(s, [])) or (j - len(isl_templates.get(s, []))) != i]
            pool[s] = own
            pred, d, _ = find_best_match(seqs[i], pool)
            c += (pred == s)
            n += 1
    print(f"    {c}/{n} ({100*c/n:.0f}%)" if n else "    (not enough samples)")
    print("    → this is what you get by keeping ISL500 + recording your own")

    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("=" * 68)
    print("""
  [A] high, [B] low   → the classifier is fine; you just sign differently
                         from the ISL500 signer. Fix: record your own samples.
  [A] low             → the signs are too similar to each other. Fix: pick
                         more visually distinct signs, or sign slower.
  [B] ≈ [C]           → adding your recordings helps. Keep doing it.
""")


def main():
    ap = argparse.ArgumentParser(description="Self-test recognition on your hands")
    ap.add_argument("--sign", action="append", default=None,
                    help="Sign(s) to test (repeatable). Default: all downloaded")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_hand_landmarks_confidence=0.5,
    )
    holistic = HolisticLandmarker.create_from_options(opts)

    # Load ISL500 templates
    from signslator.sign_to_speech import load_all_templates
    isl = load_all_templates()
    print(f"ISL500 templates loaded: {len(isl)} signs")

    signs = args.sign if args.sign else get_vocabulary()
    if not signs:
        print("No signs. Run: python run.py isl-download")
        return

    print(f"\nWill record {args.samples} sample(s) of: {', '.join(signs)}")
    print(f"{args.duration}s per sample. Press SPACE to start each.")

    sequences = {}
    for s in signs:
        print(f"\n{'='*50}\n  SIGN: {s}\n{'='*50}")
        seqs = record_samples(s, args.samples, args.duration, holistic)
        if seqs:
            sequences[s] = seqs

    holistic.close()

    if not sequences:
        print("No samples captured.")
        return

    evaluate(sequences, isl)


if __name__ == "__main__":
    main()
