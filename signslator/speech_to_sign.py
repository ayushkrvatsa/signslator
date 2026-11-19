"""
Speech → Sign and Text → Sign.

Two entry points share one lookup and one renderer:

  speech-to-sign   microphone → Whisper → text → signs → animation
  text-to-sign     typed text ────────────→ signs → animation

The animation replays the landmark sequence recorded during training. Templates
store *normalized* hand features (centred on the wrist, scaled by palm size), so
they are drawn with a fixed pixel-per-unit scale rather than being multiplied by
the canvas size — the raw values are roughly ±1, not 0..1 image coordinates.

Whisper is loaded lazily: importing this module (which `sign_chart` and
`sign_guide` do) must not pull in a 140 MB model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import queue
import time

from signslator import get_vocabulary

# ── Paths / constants ─────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "signs")
ASR_MODEL = "base"

HAND_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Canvas and projection. Normalized hand units are roughly palm-size, so one
# unit maps to ~110 px. Each hand gets its own panel.
CANVAS_W, CANVAS_H = 960, 480
PANEL_CX = (CANVAS_W // 4, 3 * CANVAS_W // 4)
PANEL_CY = CANVAS_H // 2 + 20
UNIT_PX = 110.0

# ── Whisper (lazy) ────────────────────────────────────────────────────────────

_whisper_model = None


def get_whisper_model():
    """Load Whisper on first use, not at import."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"Loading Whisper '{ASR_MODEL}' (~140 MB, first use only)...")
        _whisper_model = whisper.load_model(ASR_MODEL)
        print("Whisper loaded.")
    return _whisper_model


def transcribe(audio: np.ndarray) -> str:
    model = get_whisper_model()
    result = model.transcribe(np.asarray(audio, dtype=np.float32),
                              language="en", fp16=False)
    return result.get("text", "").strip()


# ── Microphone ────────────────────────────────────────────────────────────────

audio_queue = queue.Queue()
SAMPLE_RATE = 16000
BLOCK_DURATION = 1.0
SPEECH_TIMEOUT = 2.0        # seconds of silence that end an utterance
SILENCE_THRESHOLD = 0.01


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def record_until_silence(timeout: float = SPEECH_TIMEOUT):
    """
    Record from the default microphone until `timeout` seconds of silence.

    The stream is opened with an explicit 1-second block size — otherwise
    sounddevice delivers ~0.06 s blocks and "2 seconds of silence" ends after
    0.1 s, so the recording is truncated before the speaker finishes.

    Blocks left over from a previous recording are discarded first, so the
    second utterance never silently includes the first one's audio.
    """
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    print("Listening... (speak now)")
    frames = []
    silent_blocks = 0
    expected = max(1, int(round(timeout / BLOCK_DURATION)))
    block_samples = int(SAMPLE_RATE * BLOCK_DURATION)

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed — cannot use the microphone.")
        return None

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=block_samples,
                            callback=audio_callback):
            while silent_blocks < expected:
                try:
                    block = audio_queue.get(timeout=BLOCK_DURATION + 0.5)
                except queue.Empty:
                    break
                frames.append(block)
                energy = float(np.sqrt(np.mean(block ** 2)))
                silent_blocks = silent_blocks + 1 if energy < SILENCE_THRESHOLD else 0
    except Exception as exc:
        print(f"Microphone error: {exc}")
        return None

    if not frames:
        print("No audio captured.")
        return None

    audio = np.concatenate(frames, axis=0).flatten().astype(np.float32)
    seconds = len(audio) / SAMPLE_RATE

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak < SILENCE_THRESHOLD:
        print(f"Captured {seconds:.1f}s but it was silent — nothing to transcribe.")
        return None

    print(f"Captured {seconds:.1f}s of audio (peak {peak:.3f}).")
    if seconds < 0.3:
        print("Too short to transcribe.")
        return None
    return audio


# ── Templates ─────────────────────────────────────────────────────────────────

def load_sign_templates(sign_name: str) -> list:
    """Load every non-empty template sequence recorded for a sign."""
    sign_dir = os.path.join(DATA_DIR, sign_name.upper())
    if not os.path.isdir(sign_dir):
        return []
    templates = []
    for fname in sorted(os.listdir(sign_dir)):
        if not fname.endswith(".npy"):
            continue
        seq = np.load(os.path.join(sign_dir, fname))
        if seq.ndim == 2 and seq.shape[0] >= 5 and np.any(seq):
            templates.append(seq)
    return templates


# ── Text → sign lookup ────────────────────────────────────────────────────────

def _normalize_word(word: str) -> str:
    """Case/separator-insensitive form so GOOD_BYE matches 'good bye'."""
    return "".join(ch for ch in word.upper() if ch.isalnum())


def text_to_sign(text: str) -> list:
    """
    Map text to signs. Returns a list of (original_word, sign_or_None).

    Exact matches first, then a prefix match on at least 4 characters so
    inflections resolve ("HELLOS" → HELLO) without short words like "I"/"NO"
    absorbing unrelated vocabulary entries.
    """
    vocabulary = get_vocabulary()
    by_normalized = {_normalize_word(v): v for v in vocabulary}

    pairs = []
    for raw in text.split():
        word = raw.strip(".,!?;:'\"")
        if not word:
            continue

        upper = word.upper()
        if upper in vocabulary:
            pairs.append((word, upper))
            continue

        norm = _normalize_word(word)
        if norm in by_normalized:
            pairs.append((word, by_normalized[norm]))
            continue

        matched = None
        for vocab_word in sorted(vocabulary, key=len, reverse=True):
            if len(vocab_word) < 4:
                continue
            vnorm = _normalize_word(vocab_word)
            if norm.startswith(vnorm) or vnorm.startswith(norm):
                matched = vocab_word
                break
        pairs.append((word, matched))

    return pairs


# ── Animation ─────────────────────────────────────────────────────────────────

def _hand_points(features: np.ndarray):
    """
    Split a 144-dim feature vector into left/right hand (21, 3) arrays.

    Returns None for a hand that was not in frame. The wrist is exactly (0,0,0)
    after normalization, so presence is tested on fingertips instead.
    """
    left = features[0:63].reshape(21, 3)
    right = features[63:126].reshape(21, 3)

    def present(pts):
        return bool(np.any(pts[8]) or np.any(pts[12]) or np.any(pts[16]))

    return (left if present(left) else None,
            right if present(right) else None)


def _project(pts: np.ndarray, cx: int, cy: int) -> list:
    """Normalized hand units → canvas pixels, around a panel centre."""
    return [(int(cx + float(x) * UNIT_PX), int(cy + float(y) * UNIT_PX))
            for x, y, _z in pts]


def render_frame(left, right, label: str, sublabel: str = "") -> np.ndarray:
    """Draw one animation frame on a fresh canvas."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    for pts, cx, colour, line_colour in (
        (left, PANEL_CX[0], (255, 120, 40), (180, 80, 0)),
        (right, PANEL_CX[1], (60, 160, 255), (0, 90, 200)),
    ):
        if pts is None:
            cv2.putText(canvas, "no hand", (cx - 40, PANEL_CY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 80), 1)
            continue
        pixels = _project(pts, cx, PANEL_CY)
        for i, j in HAND_CONN:
            cv2.line(canvas, pixels[i], pixels[j], line_colour, 2)
        for p in pixels:
            cv2.circle(canvas, p, 4, colour, -1)

    cv2.putText(canvas, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (240, 240, 245), 2)
    if sublabel:
        cv2.putText(canvas, sublabel, (16, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 155, 170), 1)
    return canvas


def replay_sign_animation(sign_name: str, duration: float = 2.0,
                          window: str = "SignSlator") -> bool:
    """
    Replay a sign's recorded motion. Uses the middle template, which is usually
    the most representative of the set.
    """
    templates = load_sign_templates(sign_name)
    if not templates:
        print(f"No recording for '{sign_name}'. Train it first.")
        return False

    template = templates[len(templates) // 2]
    n = template.shape[0]
    print(f"Replaying '{sign_name}' — {n} frames over {duration:.1f}s")

    delay_ms = max(1, int(duration * 1000 / n))
    for idx in range(n):
        left, right = _hand_points(template[idx])
        frame = render_frame(
            left, right, f"Sign: {sign_name}",
            f"{len(templates)} sample(s)  ·  frame {idx + 1}/{n}")
        cv2.imshow(window, frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break
    return True


def show_text(word: str, note: str = "", window: str = "SignSlator") -> None:
    """Show an unknown word as text — there is no recording for it."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.putText(canvas, "No sign recorded for", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 155, 170), 1)
    cv2.putText(canvas, word, (16, 150), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (240, 240, 245), 3)
    if note:
        cv2.putText(canvas, note, (16, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (150, 155, 170), 1)
    cv2.imshow(window, canvas)
    cv2.waitKey(2000)


# ── Shared playback ───────────────────────────────────────────────────────────

def play_text(text: str, duration: float = 2.0, speak: bool = False) -> list:
    """
    Look up `text` and replay each sign. Returns the matched sign names.

    Works with or without a microphone — this is the text → sign path.
    """
    vocabulary = get_vocabulary()
    if not vocabulary:
        print("No signs recorded yet. Train one:  python run.py train HELLO")
        return []

    pairs = text_to_sign(text)
    if not pairs:
        print("Nothing to look up.")
        return []

    window = "SignSlator — Text to Sign"
    matched = []

    print(f"\nText: {text}")
    for word, sign in pairs:
        if sign is None:
            print(f"  {word:<14} -> no sign recorded")
            show_text(word, "train it:  python run.py train " + word.upper(),
                      window=window)
            continue

        print(f"  {word:<14} -> {sign}")
        replay_sign_animation(sign, duration=duration, window=window)
        matched.append(sign)

    if speak and matched:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(" ".join(matched))
            engine.runAndWait()
        except Exception as exc:
            print(f"(TTS unavailable: {exc})")

    cv2.destroyAllWindows()
    return matched


# ── Speech loop ───────────────────────────────────────────────────────────────

def main_loop():
    """Continuous speech → sign. Ctrl+C to stop."""
    vocabulary = get_vocabulary()

    print("=" * 55)
    print("SignSlator — Speech → Sign")
    print("Speak a word or sentence; the matching signs are replayed.")
    print("Press Ctrl+C to stop.\n")

    if vocabulary:
        print(f"Signs available ({len(vocabulary)}): {', '.join(vocabulary)}")
    else:
        print("No signs recorded yet — every word will show as text.")
        print("Train one with:  python run.py train HELLO\n")

    try:
        while True:
            audio = record_until_silence()
            if audio is None:
                continue

            text = transcribe(audio)
            if not text:
                print("(no speech recognised)\n")
                continue

            print(f"\nYou said: {text}")
            play_text(text, duration=2.0, speak=True)
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nStopped.")
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(description="SignSlator — Speech/Text → Sign")
    ap.add_argument("text", nargs="*", default=None,
                    help="Text to convert (omit to use the microphone)")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Seconds per sign (default 2.0)")
    ap.add_argument("--speak", action="store_true",
                    help="Speak the matched signs aloud")
    ap.add_argument("--list", action="store_true",
                    help="List recorded signs and exit")
    args = ap.parse_args()

    if args.list:
        vocab = get_vocabulary()
        if not vocab:
            print("No signs recorded yet.")
        else:
            print(f"\nRecorded signs ({len(vocab)}):")
            for v in vocab:
                print(f"  {v}: {len(load_sign_templates(v))} sample(s)")
        return

    if args.text:
        play_text(" ".join(args.text), duration=args.duration, speak=args.speak)
    else:
        main_loop()


if __name__ == "__main__":
    main()
