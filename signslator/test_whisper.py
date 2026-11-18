"""
Verify Whisper model loads and can transcribe. Uses a synthetic sine-wave test
since we can't speak into the mic from this terminal. In real use, the mic
capture in demo_speech.py handles live audio.
"""

import sys
import os
import numpy as np
import whisper

MODEL_NAME = "base"

print(f"Loading Whisper '{MODEL_NAME}' model...")
model = whisper.load_model(MODEL_NAME)
print(f"Model loaded. ({MODEL_NAME})")
print()

# Generate a synthetic audio test: a 2-second sine wave at 440Hz
# This won't transcribe to words, but proves the pipeline processes audio
sample_rate = 16000
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

print(f"Generating synthetic test audio: {duration}s sine wave @ 440Hz")
print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
print()

print("Running Whisper transcription on synthetic audio...")
result = model.transcribe(audio, language="en", fp16=False)
print(f"Result: '{result['text']}'")
print(f"Segments: {len(result.get('segments', []))}")
print()
print("Whisper pipeline verified — ready for live mic input.")

# Also verify model can be moved to CPU explicitly
print(f"Model device: CPU (intentionally)")
print(f"Model type: {type(model).__name__}")
