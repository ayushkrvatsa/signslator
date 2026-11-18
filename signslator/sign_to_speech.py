"""
Phase 2: Sign → Speech — Training & Recognition Engine.

How it works:
  1. TRAINING MODE: User performs a sign N times in front of camera.
     Each performance captures hand landmark sequences (21 pts × 2 hands × frames).
     Stored as templates in data/signs/{SIGN_NAME}/template_{i}.npy

  2. RECOGNITION MODE: Live camera → landmarks → compare against templates
     → best match → speak sign name via TTS.

Feature extraction:
  - Hand landmarks (21 points × 2 hands = 42 3D points per frame)
  - Relative positions (hand-to-hand distance, hand-to-body)
  - Finger extension ratios (simplified)
  - Movement velocity (frame-to-frame delta)

Classification:
  - DTW (Dynamic Time Warping) distance between live sequence and templates
  - Nearest template wins
  - Confidence threshold to avoid false positives
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)
import numpy as np
import os
import time
import json
import pyttsx3

from signslator import VOCABULARY, SIGN_LANGUAGE

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "signs")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "holistic_landmarker.task")

# ── MediaPipe setup ─────────────────────────────────────────────────────────────

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)
holistic = HolisticLandmarker.create_from_options(options)

# ── TTS ────────────────────────────────────────────────────────────────────────

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Try to use a female voice if available (Zira)
voices = tts_engine.getProperty('voices')
for v in voices:
    if 'Zira' in v.name or 'Female' in v.name:
        tts_engine.setProperty('voice', v.id)
        break

# ── Helpers ─────────────────────────────────────────────────────────────────────

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


NORMALIZE_SCALE = True


def _normalize_hand(points: np.ndarray) -> np.ndarray:
    """
    Make a hand's landmarks invariant to scale and translation.

    MediaPipe's normalized coords depend on how far the hand is from the camera
    and on the person's hand size — the two biggest reasons a template from one
    signer fails on another. Centering on the wrist and dividing by the
    wrist→middle-MCP distance removes both, leaving pure hand *shape*.

    points: (21, 3) array. Returns the same shape, normalized.
    """
    if not np.any(points):
        return points
    centered = points - points[0]              # landmark 0 = wrist
    scale = np.linalg.norm(centered[9])       # landmark 9 = middle finger MCP
    if scale < 1e-6:
        return centered
    return centered / scale


def extract_hand_features(result) -> np.ndarray:
    """
    Extract a feature vector from hand landmarks in a single frame.
    Returns a fixed-size vector (always the same length regardless of
    how many hands are detected).

    Layout (144 values):
      [0:63]    left hand  — 21 landmarks × (x, y, z), scale/translation normalized
      [63:126]  right hand — same
      [126:144] derived     — hand centers, inter-hand distance

    If a hand is missing, zeros are used.
    """
    features = np.zeros(144, dtype=np.float32)

    def grab(landmarks):
        """Turn a landmark list into a normalized (21, 3) array."""
        if not landmarks:
            return None
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if not np.any(pts):
            return None
        return _normalize_hand(pts) if NORMALIZE_SCALE else pts

    left = grab(result.left_hand_landmarks)
    right = grab(result.right_hand_landmarks)

    if left is not None:
        features[0:63] = left.flatten()
    if right is not None:
        features[63:126] = right.flatten()

    # ── Derived features ──
    # Hand centers in normalized space + inter-hand distance.
    deriv = 126
    if left is not None and right is not None:
        lc = left.mean(axis=0)
        rc = right.mean(axis=0)
        features[deriv] = np.linalg.norm(lc - rc)
        features[deriv + 1:deriv + 3] = lc[0:2]
        features[deriv + 3:deriv + 5] = rc[0:2]
    elif left is not None:
        lc = left.mean(axis=0)
        features[deriv + 1:deriv + 3] = lc[0:2]
    elif right is not None:
        rc = right.mean(axis=0)
        features[deriv + 3:deriv + 5] = rc[0:2]

    return features


def extract_sequence_features(result_list: list) -> np.ndarray:
    """Extract feature matrix from a list of HolisticLandmarkerResult objects."""
    seq = []
    for r in result_list:
        f = extract_hand_features(r)
        seq.append(f)
    return np.array(seq) if seq else np.zeros((0, 144))


def _resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a sequence to a fixed length (fast, avoids huge DTW matrices)."""
    n = seq.shape[0]
    if n == target_len:
        return seq
    if n == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim > 1 else 1))
    idx = np.linspace(0, n - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (idx - lo)[:, None]
    return seq[lo] * (1 - frac) + seq[hi] * frac


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, max_len: int = 64) -> float:
    """
    Compute DTW distance between two sequences (vectorized cost matrix).

    Sequences are resampled to at most `max_len` frames first — this keeps the
    O(n*m) accumulation cheap while preserving the motion shape.
    Returns the normalized distance (lower = more similar).
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float('inf')

    # Downsample long sequences to bound the matrix size
    if n > max_len:
        seq1 = _resample(seq1, max_len)
        n = max_len
    if m > max_len:
        seq2 = _resample(seq2, max_len)
        m = max_len

    # Vectorized cost matrix: pairwise Euclidean distances
    diff = seq1[:, None, :] - seq2[None, :, :]      # (n, m, d)
    cost = np.sqrt((diff * diff).sum(axis=2))       # (n, m)

    # DTW accumulation (sequential, but over a small matrix now)
    dtw = np.empty((n, m), dtype=np.float64)
    dtw[0, 0] = cost[0, 0]
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + cost[i, 0]
    for j in range(1, m):
        dtw[0, j] = dtw[0, j - 1] + cost[0, j]
    for i in range(1, n):
        prev = dtw[i - 1]
        cur = dtw[i]
        ci = cost[i]
        for j in range(1, m):
            cur[j] = ci[j] + min(prev[j], cur[j - 1], prev[j - 1])

    # Normalize by path length
    return dtw[n - 1, m - 1] / (n + m)


def find_best_match(live_seq: np.ndarray, templates: dict) -> tuple:
    """
    Compare live sequence against all templates.
    Returns (best_sign_name, best_distance, all_distances).
    """
    if live_seq.shape[0] < 5:  # too short
        return None, float('inf'), {}

    distances = {}
    for sign_name, seq_list in templates.items():
        min_dist = float('inf')
        for template_seq in seq_list:
            if template_seq.shape[0] < 5:
                continue
            # Use a window of the live sequence for comparison
            d = dtw_distance(live_seq, template_seq)
            if d < min_dist:
                min_dist = d
        distances[sign_name] = min_dist

    if not distances:
        return None, float('inf'), {}

    best_sign = min(distances, key=distances.get)
    best_dist = distances[best_sign]

    return best_sign, best_dist, distances


def calibrate_thresholds(templates: dict, verbose: bool = True) -> tuple:
    """
    Auto-calibrate (max_distance, margin_ratio) from the template set itself.

    Uses leave-one-out distances to learn what a "good" match looks like:
      - correct_max  = worst distance when matching a held-out sample of the same sign
      - wrong_min    = best distance when matching a different sign (the confusion floor)

    Returns (max_distance, margin_ratio):
      - max_distance: accept a match only if its distance is below this
      - margin_ratio: and only if it beats the runner-up by at least this factor
    """
    same_dists = []   # distances to same-sign templates
    diff_dists = []   # distances to other-sign templates

    for sign, tlist in templates.items():
        for i, held in enumerate(tlist):
            # Resample the held-out sequence once
            held_r = _resample(held, min(len(held), 48)) if len(held) > 48 else held
            for other_sign, olist in templates.items():
                for j, other in enumerate(olist):
                    if other_sign == sign and j == i:
                        continue
                    d = dtw_distance(held_r, other)
                    if other_sign == sign:
                        same_dists.append(d)
                    else:
                        diff_dists.append(d)

    if not same_dists or not diff_dists:
        return 5.0, 1.0

    same_dists = np.array(same_dists)
    diff_dists = np.array(diff_dists)

    # Accept threshold: comfortably above typical same-sign distance,
    # below the typical different-sign distance.
    same_high = float(np.percentile(same_dists, 90))
    diff_low = float(np.percentile(diff_dists, 10))

    max_distance = same_high * 1.5
    if max_distance >= diff_low:
        # Overlapping distributions — sit in the middle
        max_distance = (same_high + diff_low) / 2.0

    # Margin: how much better than the runner-up
    margin_ratio = 1.15 if diff_low > same_high else 1.0

    if verbose:
        print(f"  Calibration: same-sign dist mean={same_dists.mean():.3f} p90={same_high:.3f}")
        print(f"               diff-sign dist mean={diff_dists.mean():.3f} p10={diff_low:.3f}")
        print(f"               → max_distance={max_distance:.3f}, margin_ratio={margin_ratio:.2f}")

    return max_distance, margin_ratio


# ── Training ────────────────────────────────────────────────────────────────────

def train_sign(sign_name: str, num_samples: int = 5, duration: float = 3.0):
    """
    Train: capture `num_samples` examples of `sign_name`.
    Each example records landmarks for `duration` seconds.
    """
    sign_dir = os.path.join(DATA_DIR, sign_name.upper())
    os.makedirs(sign_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"TRAINING: '{sign_name}'")
    print(f"Capturing {num_samples} samples × {duration}s each")
    print(f"Press SPACE to start each sample, 'q' to cancel")
    print(f"{'='*55}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for sample_i in range(num_samples):
        print(f"Sample {sample_i + 1}/{num_samples}: Perform '{sign_name}' now.")
        print("Press SPACE to start recording...")

        # Wait for spacebar
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Press SPACE to record '{sign_name}' (sample {sample_i+1})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SignSlator — Training", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q') or key == 27:
                print("Training cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return False

        # Record
        print(f"Recording sample {sample_i + 1}... GO!")
        results = []
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = holistic.detect_for_video(mp_image, int(time.time() * 1000))
            results.append(result)

            # Visual feedback
            display = frame.copy()
            progress = (time.time() - start_time) / duration
            cv2.rectangle(display, (0, 0), (int(640 * progress), 10), (0, 255, 0), -1)
            cv2.putText(display, f"Recording... {frame_count} frames",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("SignSlator — Training", display)
            cv2.waitKey(1)

        # Save
        seq = extract_sequence_features(results)
        if seq.shape[0] >= 5:
            template_path = os.path.join(sign_dir, f"template_{sample_i}.npy")
            np.save(template_path, seq)
            print(f"  Saved: {template_path} ({seq.shape[0]} frames)")
        else:
            print(f"  WARNING: Too few frames ({seq.shape[0]}), skipping")

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTraining complete for '{sign_name}'. {num_samples} samples saved.")
    return True


# ── Recognition ────────────────────────────────────────────────────────────────

def recognize_sign(confidence_threshold: float = 0.3,
                   max_distance: float = None,
                   window_seconds: float = 2.0) -> str:
    """
    Live recognition loop.
    Returns the recognized sign name, or None if no confident match.

    If max_distance is None, it is auto-calibrated from the template set.
    """
    # Load all templates
    templates = load_all_templates()
    if not templates:
        print("No trained signs found. Train some signs first.")
        return None

    # Auto-calibrate thresholds from the templates themselves
    print("\nCalibrating thresholds from templates...")
    auto_max_dist, margin_ratio = calibrate_thresholds(templates)
    if max_distance is None:
        max_distance = auto_max_dist

    print(f"Recognizing... ({len(templates)} signs loaded)")
    print(f"Accept threshold: distance < {max_distance:.3f} and "
          f"beats runner-up by {margin_ratio:.2f}×")
    print("Press 'q' to quit, 't' to go back to training\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Ring buffer for recent frames
    buffer_size = int(30 * window_seconds)  # 30 fps × window_seconds
    frame_buffer = []
    result_buffer = []

    last_spoken = None
    cooldown = 3.0  # seconds between TTS announcements
    last_tts_time = 0

    running = True
    while running:
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

        # Try recognition on the buffer
        if len(result_buffer) >= 15:  # at least 0.5 seconds
            live_seq = extract_sequence_features(result_buffer)
            best_sign, best_dist, all_dists = find_best_match(live_seq, templates)

            display = frame.copy()

            # Show all distances
            y = 50
            for sign_name, dist in sorted(all_dists.items(), key=lambda x: x[1]):
                color = (0, 255, 0) if sign_name == best_sign else (100, 100, 100)
                cv2.putText(display, f"{sign_name}: {dist:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 18

            # Highlight best match — require distance AND margin over runner-up
            runner_up_dist = None
            for sign_name, dist in sorted(all_dists.items(), key=lambda x: x[1]):
                if sign_name != best_sign:
                    runner_up_dist = dist
                    break

            margin_ok = True
            if runner_up_dist is not None and runner_up_dist > 0:
                margin_ok = (runner_up_dist / best_dist) >= margin_ratio if best_dist > 0 else True

            if best_sign and best_dist < max_distance and margin_ok:
                confidence = max(0, 1.0 - best_dist / max_distance)
                label = f"→ {best_sign} ({confidence:.0%})"
                cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                # Speak if new sign and cooldown passed
                now = time.time()
                if best_sign != last_spoken and now - last_tts_time > cooldown:
                    print(f"Recognizing: {best_sign} (confidence: {confidence:.0%}, "
                          f"dist={best_dist:.3f})")
                    tts_engine.say(best_sign)
                    tts_engine.runAndWait()
                    last_spoken = best_sign
                    last_tts_time = now
            elif best_sign and best_dist < max_distance and not margin_ok:
                cv2.putText(display, f"→ {best_sign}? (ambiguous vs {runner_up_dist:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            else:
                cv2.putText(display, "→ No confident match", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw landmarks on display
            _draw_landmarks(display, result)

        else:
            display = frame.copy()
            cv2.putText(display, "Warming up...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SignSlator — Recognition", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False
        elif key == ord('t'):
            running = False

    cap.release()
    cv2.destroyAllWindows()
    return last_spoken


def _draw_landmarks(display, result, show_face=False):
    """Draw hand landmarks on display frame. Body pose is detected but not shown."""
    # Hands only — blue dots + connections
    all_hands = (result.left_hand_landmarks or []) + (result.right_hand_landmarks or [])
    for idx, lm in enumerate(all_hands):
        x, y = int(lm.x * display.shape[1]), int(lm.y * display.shape[0])
        cv2.circle(display, (x, y), 4, (255, 0, 0), -1)

    # Draw hand connections (indices 0-20 = left hand, 21+ = right hand)
    left_count = len(result.left_hand_landmarks or [])
    for i, j in HAND_CONN:
        # Left hand connections
        if i < left_count and j < left_count:
            li, lj = all_hands[i], all_hands[j]
            cv2.line(display, (int(li.x*display.shape[1]), int(li.y*display.shape[0])),
                     (int(lj.x*display.shape[1]), int(lj.y*display.shape[0])),
                     (255, 100, 0), 2)
        # Right hand connections (offset by left hand count)
        elif left_count > 0 and (i + left_count) < len(all_hands) and (j + left_count) < len(all_hands):
            ri, rj = all_hands[i + left_count], all_hands[j + left_count]
            cv2.line(display, (int(ri.x*display.shape[1]), int(ri.y*display.shape[0])),
                     (int(rj.x*display.shape[1]), int(rj.y*display.shape[0])),
                     (255, 100, 0), 2)

    # Optional: face landmarks (for signs that use facial expression)
    if show_face and result.face_landmarks:
        for lm in result.face_landmarks:
            x, y = int(lm.x * display.shape[1]), int(lm.y * display.shape[0])
            cv2.circle(display, (x, y), 1, (0, 255, 255), -1)


# ── Template management ────────────────────────────────────────────────────────

def load_all_templates() -> dict:
    """Load all trained sign templates from data/signs/."""
    templates = {}
    if not os.path.exists(DATA_DIR):
        return templates

    for sign_name in os.listdir(DATA_DIR):
        sign_dir = os.path.join(DATA_DIR, sign_name)
        if not os.path.isdir(sign_dir):
            continue
        templates[sign_name] = []
        for fname in sorted(os.listdir(sign_dir)):
            if fname.endswith('.npy'):
                path = os.path.join(sign_dir, fname)
                seq = np.load(path)
                templates[sign_name].append(seq)
        if templates[sign_name]:
            print(f"  Loaded '{sign_name}': {len(templates[sign_name])} templates")
        else:
            del templates[sign_name]

    return templates


def list_trained_signs():
    """Print list of trained signs and their template counts."""
    if not os.path.exists(DATA_DIR):
        print("No trained signs yet.")
        return
    print("\nTrained signs:")
    for sign_name in sorted(os.listdir(DATA_DIR)):
        sign_dir = os.path.join(DATA_DIR, sign_name)
        if os.path.isdir(sign_dir):
            count = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
            print(f"  {sign_name}: {count} templates")


# ── Main CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SignSlator — Sign → Speech")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a sign")
    train_parser.add_argument("sign_name", nargs="?", default=None,
                              help=f"Sign name (default: prompt). Options: {', '.join(VOCABULARY)}")
    train_parser.add_argument("--samples", type=int, default=5,
                              help="Number of samples to capture (default: 5)")
    train_parser.add_argument("--duration", type=float, default=3.0,
                              help="Seconds per sample (default: 3.0)")

    # Recognize
    rec_parser = subparsers.add_parser("recognize", help="Live sign recognition")
    rec_parser.add_argument("--threshold", type=float, default=0.3,
                           help="Confidence threshold (default: 0.3)")
    rec_parser.add_argument("--max-dist", type=float, default=5.0,
                           help="Max DTW distance for match (default: 5.0)")
    rec_parser.add_argument("--window", type=float, default=2.0,
                           help="Recognition window in seconds (default: 2.0)")

    # List
    subparsers.add_parser("list", help="List trained signs")

    args = parser.parse_args()

    if args.command == "train":
        sign_name = args.sign_name
        if sign_name is None:
            print(f"\nAvailable signs to train:")
            for i, s in enumerate(VOCABULARY, 1):
                print(f"  {i}. {s}")
            print(f"\nOr enter a custom sign name.")
            sign_name = input("\nWhich sign to train? ").strip().upper()
            if not sign_name:
                print("Cancelled.")
                return
        train_sign(sign_name, args.samples, args.duration)
        list_trained_signs()

    elif args.command == "recognize":
        recognize_sign(args.threshold, args.max_dist, args.window)

    elif args.command == "list":
        list_trained_signs()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
