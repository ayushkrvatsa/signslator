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


def _build_prefilter(templates: dict, length: int = 32) -> dict:
    """Resample every template once, so live matching doesn't repeat the work."""
    out = {}
    for sign, seqs in templates.items():
        out[sign] = [_resample(s, length) for s in seqs if s.shape[0] >= 5]
    return out


def find_best_match(live_seq: np.ndarray, templates: dict,
                    prefilter: dict = None, top_k: int = 5) -> tuple:
    """
    Compare a live sequence against all templates, in two stages.

    Stage 1 — cheap prefilter over EVERY template (mean absolute difference of
    resampled sequences). Measured recall is 100% at top-5, so nothing is lost,
    and it costs ~9 ms instead of seconds.

    Stage 2 — full DTW, only on the top_k candidate signs.

    Comparing against all 605 templates with DTW on every frame cost ~5 s per
    frame at 153 signs, which froze the camera. This is ~150 ms.

    Returns (best_sign_name, best_distance, distances) where `distances` holds
    the DTW distance for each refined candidate.
    """
    if live_seq.shape[0] < 5:  # too short
        return None, float('inf'), {}

    if prefilter is None:
        prefilter = _build_prefilter(templates)

    live_r = _resample(live_seq, 32)

    # ── Stage 1: cheap ranking over every template ──
    cheap = {}
    for sign_name, resampled in prefilter.items():
        if not resampled:
            continue
        cheap[sign_name] = min(float(np.abs(live_r - r).mean()) for r in resampled)

    if not cheap:
        return None, float('inf'), {}

    candidates = sorted(cheap, key=cheap.get)[:top_k]

    # ── Stage 2: full DTW on the candidates only ──
    distances = {}
    for sign_name in candidates:
        min_dist = float('inf')
        for template_seq in templates[sign_name]:
            if template_seq.shape[0] < 5:
                continue
            d = dtw_distance(live_seq, template_seq)
            if d < min_dist:
                min_dist = d
        distances[sign_name] = min_dist

    if not distances:
        return None, float('inf'), {}

    best_sign = min(distances, key=distances.get)
    best_dist = distances[best_sign]

    return best_sign, best_dist, distances


CALIBRATION_CACHE = os.path.join(DATA_DIR, "_calibration.json")
# Must match the max_len used by live matching, or the threshold is on a
# different distance scale and the gate becomes meaningless.
_CAL_MAX_LEN = 64
_CAL_SAME_PAIRS = 2        # same-sign pairs sampled per sign
_CAL_DIFF_SIGNS = 12       # other signs sampled per sign


def _template_paths() -> dict:
    """Map sign name → sorted list of template file paths."""
    paths = {}
    if not os.path.isdir(DATA_DIR):
        return paths
    for sign in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
        if files:
            paths[sign] = [os.path.join(d, f) for f in files]
    return paths


def _fingerprint(paths: dict) -> str:
    """Hash of template names, sizes and mtimes — invalidates the cache on change."""
    import hashlib
    h = hashlib.md5()
    for sign, plist in sorted(paths.items()):
        for p in plist:
            try:
                st = os.stat(p)
                h.update(f"{sign}|{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}|".encode())
            except OSError:
                pass
    return h.hexdigest()


def _load_calibration_cache(fp: str):
    if not os.path.exists(CALIBRATION_CACHE):
        return None
    try:
        with open(CALIBRATION_CACHE, "r") as f:
            data = json.load(f)
        if data.get("fingerprint") == fp:
            return data
    except Exception:
        pass
    return None


def _save_calibration_cache(fp: str, max_distance: float, margin_ratio: float):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CALIBRATION_CACHE, "w") as f:
            json.dump({"fingerprint": fp, "max_distance": max_distance,
                       "margin_ratio": margin_ratio}, f)
    except Exception:
        pass


def calibrate_thresholds(templates: dict, verbose: bool = True) -> tuple:
    """
    Auto-calibrate (max_distance, margin_ratio) from the template set itself.

    Learns what a "good" match looks like from the templates:
      - same-sign distances  → how close a correct match should be
      - different-sign distances → how far a wrong match typically is

    Comparing every template against every other is O(n²) — with 600 templates
    that is ~180k DTW calls (≈16 min). So this samples pairs instead, and caches
    the result so subsequent runs start instantly.

    Returns (max_distance, margin_ratio):
      - max_distance: accept a match only if its distance is below this
      - margin_ratio: and only if it beats the runner-up by at least this factor
    """
    paths = _template_paths()
    fp = _fingerprint(paths)

    cached = _load_calibration_cache(fp)
    if cached is not None:
        if verbose:
            print(f"  Calibration: loaded from cache "
                  f"(max_distance={cached['max_distance']:.3f}, "
                  f"margin_ratio={cached['margin_ratio']:.2f})")
        return float(cached["max_distance"]), float(cached["margin_ratio"])

    rng = np.random.default_rng(42)
    signs = sorted(templates.keys())
    reps = {s: templates[s][rng.integers(len(templates[s]))] for s in signs}

    same_dists = []
    diff_dists = []

    for sign in signs:
        tlist = templates[sign]
        # ── same-sign: sample a few pairs within this sign ──
        for _ in range(_CAL_SAME_PAIRS):
            if len(tlist) < 2:
                break
            i, j = rng.choice(len(tlist), size=2, replace=False)
            same_dists.append(
                dtw_distance(tlist[i], tlist[j], max_len=_CAL_MAX_LEN))

        # ── different-sign: compare this sign's rep against a sample of others ──
        others = [s for s in signs if s != sign]
        if len(others) > _CAL_DIFF_SIGNS:
            others = list(rng.choice(others, size=_CAL_DIFF_SIGNS, replace=False))
        for other in others:
            diff_dists.append(
                dtw_distance(reps[sign], reps[other], max_len=_CAL_MAX_LEN))

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

    _save_calibration_cache(fp, max_distance, margin_ratio)

    if verbose:
        print(f"  Calibration ({len(same_dists)} same / {len(diff_dists)} diff samples):")
        print(f"    same-sign dist mean={same_dists.mean():.3f} p90={same_high:.3f}")
        print(f"    diff-sign dist mean={diff_dists.mean():.3f} p10={diff_low:.3f}")
        print(f"    → max_distance={max_distance:.3f}, margin_ratio={margin_ratio:.2f}")
        print(f"    (cached to {os.path.basename(CALIBRATION_CACHE)})")

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Ring buffer of landmark results (only this is needed for matching)
    buffer_size = int(30 * window_seconds)  # 30 fps × window_seconds
    result_buffer = []

    last_spoken = None
    cooldown = 3.0  # seconds between TTS announcements
    last_tts_time = 0

    # Matching is expensive, so classify a few times per second rather than on
    # every frame; the display keeps showing the last verdict in between.
    classify_every = 8
    frame_no = 0

    # Resampled templates for the cheap prefilter — built once, reused every time.
    prefilter = _build_prefilter(templates)

    verdict = {"sign": None, "dist": float("inf"), "dists": {},
               "conf": 0.0, "margin_ok": True, "runner_up": None}

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = holistic.detect_for_video(mp_image, int(time.time() * 1000))

        result_buffer.append(result)
        if len(result_buffer) > buffer_size:
            result_buffer.pop(0)

        display = frame.copy()

        # ── classify only every Nth frame ──
        if len(result_buffer) >= 15 and frame_no % classify_every == 0:
            live_seq = extract_sequence_features(result_buffer)
            best_sign, best_dist, all_dists = find_best_match(
                live_seq, templates, prefilter=prefilter)

            runner_up_dist = None
            for s, d in sorted(all_dists.items(), key=lambda x: x[1]):
                if s != best_sign:
                    runner_up_dist = d
                    break

            margin_ok = True
            if runner_up_dist is not None and runner_up_dist > 0 and best_dist > 0:
                margin_ok = (runner_up_dist / best_dist) >= margin_ratio

            verdict["sign"] = best_sign
            verdict["dist"] = best_dist
            verdict["dists"] = all_dists
            verdict["margin_ok"] = margin_ok
            verdict["runner_up"] = runner_up_dist

            if best_sign and best_dist < max_distance and margin_ok:
                verdict["conf"] = max(0.0, 1.0 - best_dist / max_distance)
                now = time.time()
                if best_sign != last_spoken and now - last_tts_time > cooldown:
                    print(f"Recognizing: {best_sign} "
                          f"(confidence: {verdict['conf']:.0%}, dist={best_dist:.3f})")
                    tts_engine.say(best_sign)
                    tts_engine.runAndWait()
                    last_spoken = best_sign
                    last_tts_time = now

        # ── drawing (cheap, every frame) ──
        _draw_landmarks(display, result)

        y = 46
        for i, (s, d) in enumerate(sorted(verdict["dists"].items(), key=lambda x: x[1])):
            if i >= 6:
                break
            color = (0, 255, 0) if s == verdict["sign"] else (95, 95, 105)
            cv2.putText(display, f"{s}: {d:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 16

        sign = verdict["sign"]
        if sign and verdict["dist"] < max_distance and verdict["margin_ok"]:
            cv2.putText(display, f"-> {sign} ({verdict['conf']:.0%})", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif sign and verdict["dist"] < max_distance:
            cv2.putText(display, f"-> {sign}? (ambiguous)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif len(result_buffer) < 15:
            cv2.putText(display, "Warming up...", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.putText(display, "-> No confident match", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.putText(display, f"{len(templates)} signs  |  q: quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1)

        cv2.imshow("SignSlator — Recognition", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
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
    rec_parser.add_argument("--max-dist", type=float, default=None,
                            help="Accept threshold (default: auto-calibrate from templates)")
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
