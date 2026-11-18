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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "signs")

# Sign language: ISL (Indian Sign Language)
SIGN_LANGUAGE = "ISL"

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
