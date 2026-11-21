"""
SignSlator — ISL Sign Language ↔ Speech bidirectional system.

Architecture:
  Sign → Speech: camera → MediaPipe landmarks → classifier → TTS
  Speech → Sign: mic → Whisper ASR → text → sign lookup → hand animation replay

Phases:
  1. Foundation: camera feed + hand landmark overlay (live proof it works)
  2. Sign→Speech: train mode + recognize mode for a controlled vocabulary
  3. Speech→Sign: Whisper ASR + landmark animation replay
  4. Combined UI
"""

import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "signs")

# Sign language: ISL (Indian Sign Language)
SIGN_LANGUAGE = "ISL"


def canonicalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Move a lone hand into the first feature block, then rebuild derived features.

    People sign one-handed signs with whichever hand is convenient. If the
    feature slot depends on which hand was used, the same sign recorded with
    the left and the right hand produces completely different vectors —
    measured at 4.3 vs 1.4 DTW distance for identical signs.

    Applied both when loading stored templates and when extracting live
    features, so recordings made before this change stay comparable.

    Layout: [0:63] hand A, [63:126] hand B, [126:144] derived.
    """
    out = np.array(seq, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] < 144:
        return out

    has_a = bool(np.any(out[:, 0:63]))
    has_b = bool(np.any(out[:, 63:126]))
    if has_b and not has_a:
        out[:, 0:63] = seq[:, 63:126]
        out[:, 63:126] = 0.0

    n = out.shape[0]
    a = out[:, 0:63].reshape(n, 21, 3)
    b = out[:, 63:126].reshape(n, 21, 3)
    pa = np.any(a, axis=(1, 2))
    pb = np.any(b, axis=(1, 2))

    ca = np.zeros((n, 3), dtype=np.float32)
    cb = np.zeros((n, 3), dtype=np.float32)
    ca[pa] = a[pa].mean(axis=1)
    cb[pb] = b[pb].mean(axis=1)

    deriv = np.zeros((n, 18), dtype=np.float32)
    both = pa & pb
    deriv[both, 0] = np.linalg.norm(ca[both] - cb[both], axis=1)
    deriv[pa, 1:3] = ca[pa, 0:2]
    deriv[pb, 3:5] = cb[pb, 0:2]
    out[:, 126:144] = deriv
    return out

# Starter vocabulary to fetch from ISL500 — every word verified to exist in
# the dataset. (ISL500 has no "Sorry"/"Thank"/"Stop" — use Regret/Thankyou/Stay.)
STARTER_WORDS = [
    # greetings & politeness
    "Hello", "Goodmorning", "Goodevening", "Goodnight", "Thankyou",
    "Please", "Regret", "Yes", "No", "Good", "Bad",
    # needs
    "Water", "Eat", "Food", "Drink", "Help", "Medicine", "Doctor",
    "Hot", "Cold", "Big", "Small", "Finish",
    # people
    "I", "You", "He", "She", "We", "They", "Mother", "Father",
    "Sister", "Family",
    # places & things
    "House", "School", "Work", "Book", "Pen", "Chair", "Door", "Window",
    # time
    "Today", "Tomorrow", "Yesterday", "Morning", "Evening", "Night",
    # questions  (ISL500 has no "When" — Which is the closest)
    "What", "Where", "Which", "Why",
]


def get_vocabulary() -> list:
    """
    Return the signs that are actually trained/downloaded, sorted.

    Single source of truth for both directions:
      - sign → speech: which signs the classifier can output
      - speech → sign: which words have an animation to replay
    """
    if not os.path.isdir(DATA_DIR):
        return []
    signs = []
    for name in os.listdir(DATA_DIR):
        sign_dir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(sign_dir):
            continue
        if any(f.endswith(".npy") for f in os.listdir(sign_dir)):
            signs.append(name.upper())
    return sorted(signs)


# Backwards-compatible alias, evaluated at import time. For a live view after
# downloading new signs, call get_vocabulary() instead.
VOCABULARY = get_vocabulary()
