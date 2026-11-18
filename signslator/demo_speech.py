"""
Phase 1B: Speech → Text via OpenAI Whisper (offline, local).
Proves: mic audio → Whisper → recognized text, fully offline.

Usage:
  python demo_speech.py           # listens once, prints result
  python demo_speech.py --loop   # continuous listening loop
"""

import argparse
import sys
import os

# Add project root so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import json

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "base"          # Options: tiny, base, small, medium, large
SAMPLE_RATE = 16000          # Whisper expects 16kHz
CHANNELS = 1
BLOCK_DURATION = 1.0         # seconds per audio block
SPEECH_TIMEOUT = 2.0         # seconds of silence before finalizing

# ── Load model (done once, cached in memory) ──────────────────────────────────

print(f"Loading Whisper model '{MODEL_NAME}'... (one-time, ~140MB)")
model = whisper.load_model(MODEL_NAME)
print(f"Model loaded. Ready.\n")

# ── Audio capture ─────────────────────────────────────────────────────────────

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block."""
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def record_until_silence(timeout=SPEECH_TIMEOUT):
    """
    Record audio blocks until silence is detected for `timeout` seconds.
    Returns concatenated audio as numpy array.
    """
    print("Listening... (speak now)")
    frames = []
    silent_blocks = 0
    expected_blocks = int(timeout / BLOCK_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        callback=audio_callback):
        while silent_blocks < expected_blocks:
            try:
                block = audio_queue.get(timeout=BLOCK_DURATION + 1)
                frames.append(block)
                # Simple energy-based voice activity detection
                energy = np.sqrt(np.mean(block ** 2))
                if energy < 0.01:  # threshold — adjust if needed
                    silent_blocks += 1
                else:
                    silent_blocks = 0
            except queue.Empty:
                break

    if not frames:
        print("No audio captured.")
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    return audio


def transcribe(audio):
    """Run Whisper on audio array."""
    result = model.transcribe(audio, language="en", fp16=False)
    return result["text"].strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def single_transcribe():
    audio = record_until_silence()
    if audio is None:
        return
    text = transcribe(audio)
    print(f"\n{'='*40}")
    print(f"YOU SAID: {text}")
    print(f"{'='*40}\n")
    return text


def loop_transcribe():
    print("Continuous mode. Press Ctrl+C to stop.\n")
    try:
        while True:
            audio = record_until_silence()
            if audio is None:
                continue
            text = transcribe(audio)
            print(f"\nYOU SAID: {text}\n")
            # Here we'd feed text into the sign lookup — Phase 3
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper speech→text demo")
    parser.add_argument("--loop", action="store_true", help="Continuous listening")
    args = parser.parse_args()

    if args.loop:
        loop_transcribe()
    else:
        single_transcribe()
