"""
Phase 3: Speech → Sign — Speech recognition + sign animation replay.

Pipeline:
  1. Mic → Whisper ASR → text
  2. Text → word lookup in sign vocabulary
  3. Sign found → replay stored landmark animation (from training data)
  4. Sign not found → show text, offer to train it

The "dummy" hand animation replays the landmark sequences recorded during
Phase 2 training, drawing hands on screen as moving dots+lines.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import whisper
import sounddevice as sd
import queue
import time
import pyttsx3
import json

from signslator import get_vocabulary, SIGN_LANGUAGE

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "signs")
ASR_MODEL = "base"

# ── Whisper ────────────────────────────────────────────────────────────────────

print(f"Loading Whisper '{ASR_MODEL}' for speech recognition...")
whisper_model = whisper.load_model(ASR_MODEL)
print("Whisper loaded.\n")

# ── TTS ────────────────────────────────────────────────────────────────────────

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.8)

# ── Constants ───────────────────────────────────────────────────────────────────

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

# ── Audio capture ──────────────────────────────────────────────────────────────

audio_queue = queue.Queue()
SAMPLE_RATE = 16000
BLOCK_DURATION = 1.0
SPEECH_TIMEOUT = 2.0  # seconds of silence before finalizing


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def record_until_silence(timeout=SPEECH_TIMEOUT):
    """Record audio until silence, return numpy array."""
    print("Listening... (speak now)")
    frames = []
    silent_blocks = 0
    expected_blocks = int(timeout / BLOCK_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        callback=audio_callback):
        while silent_blocks < expected_blocks:
            try:
                block = audio_queue.get(timeout=BLOCK_DURATION + 1)
                frames.append(block)
                energy = np.sqrt(np.mean(block ** 2))
                if energy < 0.01:
                    silent_blocks += 1
                else:
                    silent_blocks = 0
            except queue.Empty:
                break

    if not frames:
        print("No audio captured.")
        return None

    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio):
    """Whisper transcription."""
    result = whisper_model.transcribe(audio, language="en", fp16=False)
    return result["text"].strip()


# ── Sign animation replay ──────────────────────────────────────────────────────

def load_sign_templates(sign_name: str) -> list:
    """Load all template sequences for a sign."""
    sign_dir = os.path.join(DATA_DIR, sign_name.upper())
    if not os.path.exists(sign_dir):
        return []
    templates = []
    for fname in sorted(os.listdir(sign_dir)):
        if fname.endswith('.npy'):
            templates.append(np.load(os.path.join(sign_dir, fname)))
    return templates


def replay_sign_animation(sign_name: str, duration: float = 3.0):
    """
    Replay a sign's landmark animation on screen.
    Uses the first available template for the sign.
    Draws hands moving on screen as blue dots+lines.
    """
    templates = load_sign_templates(sign_name)
    if not templates:
        print(f"No animation data for '{sign_name}'. Train it first.")
        return False

    template = templates[0]  # Use first template
    print(f"Replaying '{sign_name}' animation ({template.shape[0]} frames)")

    # Create a black display
    display = np.zeros((480, 640, 3), dtype=np.uint8)

    # Normalize frame timing
    frame_delay = duration / template.shape[0] if template.shape[0] > 0 else 0.05

    for frame_idx in range(template.shape[0]):
        display.fill(0)

        # Get features for this frame
        features = template[frame_idx]

        # Reconstruct hand landmarks from features
        left_lms = []
        right_lms = []

        for i in range(21):
            lx = features[i * 3]
            ly = features[i * 3 + 1]
            lz = features[i * 3 + 2]
            if lx != 0 or ly != 0:
                left_lms.append((int(lx * 640), int(ly * 480)))

        for i in range(21):
            rx = features[63 + i * 3]
            ry = features[63 + i * 3 + 1]
            rz = features[63 + i * 3 + 2]
            if rx != 0 or ry != 0:
                right_lms.append((int(rx * 640), int(ry * 480)))

        # Draw left hand
        for x, y in left_lms:
            cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
        for i, j in HAND_CONN:
            if i < len(left_lms) and j < len(left_lms):
                cv2.line(display, left_lms[i], left_lms[j], (255, 100, 0), 2)

        # Draw right hand
        offset_x = 320  # center the right hand
        for x, y in right_lms:
            cv2.circle(display, (x + offset_x, y), 5, (0, 0, 255), -1)
        for i, j in HAND_CONN:
            if i < len(right_lms) and j < len(right_lms):
                p1 = (right_lms[i][0] + offset_x, right_lms[i][1])
                p2 = (right_lms[j][0] + offset_x, right_lms[j][1])
                cv2.line(display, p1, p2, (100, 0, 255), 2)

        # Label
        cv2.putText(display, f"Replaying: {sign_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Frame {frame_idx + 1}/{template.shape[0]}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("SignSlator — Speech → Sign", display)
        cv2.waitKey(int(frame_delay * 1000))

    cv2.destroyAllWindows()
    return True


# ── Text → Sign lookup ──────────────────────────────────────────────────────────

def text_to_sign(text: str) -> list:
    """
    Convert transcribed text to a list of sign names to display.
    Word-by-word lookup against the signs that actually have templates.
    Returns None for words with no sign available.
    """
    vocabulary = get_vocabulary()          # live view of downloaded signs
    words = text.upper().split()
    signs_to_show = []

    for word in words:
        if word in vocabulary:
            signs_to_show.append(word)
            continue

        # Fuzzy: only allow a vocabulary word to absorb a clear inflection
        # (e.g. "HELLOS" → HELLO, "WATER." → WATER). Require a prefix match on
        # at least 4 characters so short words like "I"/"NO" don't match anything.
        matched = None
        for vocab_word in vocabulary:
            if len(vocab_word) < 4:
                continue
            if word.startswith(vocab_word) or vocab_word.startswith(word):
                matched = vocab_word
                break
        signs_to_show.append(matched)      # None when nothing matches

    return signs_to_show


# ── Main loop ───────────────────────────────────────────────────────────────────

def main_loop():
    """Continuous speech → sign loop."""
    print("=" * 55)
    print("SignSlator — Speech → Sign")
    print("Speak a word/sentence, see the sign animation")
    print("Press Ctrl+C to stop\n")

    # Check what signs are available
    available = get_vocabulary()

    if available:
        print(f"Available signs for animation ({len(available)}): {', '.join(available)}")
    else:
        print("No signs available for animation.")
        print("Get some first:  python run.py train HELLO")
        print("Starting anyway — unknown words will show as text.\n")

    try:
        while True:
            # Listen
            audio = record_until_silence()
            if audio is None:
                continue

            # Transcribe
            text = transcribe(audio)
            if not text:
                print("(silence)")
                continue

            print(f"\n{'='*40}")
            print(f"YOU SAID: {text}")
            print(f"{'='*40}")

            # Speak it back
            tts_engine.say(text)
            tts_engine.runAndWait()

            # Convert to signs
            signs = text_to_sign(text)
            print(f"Signs to show: {signs}")

            # Replay each sign
            for sign_name in signs:
                if sign_name is None:
                    print(f"  → Unknown word, showing as text")
                    # Show text on screen
                    display = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display, text,
                                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 255, 255), 2)
                    cv2.imshow("SignSlator — Speech → Sign", display)
                    cv2.waitKey(2000)
                else:
                    print(f"  → Replaying '{sign_name}'")
                    replay_sign_animation(sign_name, duration=2.0)

            # Small pause between iterations
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main_loop()
