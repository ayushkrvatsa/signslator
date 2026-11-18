#!/usr/bin/env python3
"""
SignSlator setup — one command to make the project runnable on any machine.

What it does:
  1. Checks Python version
  2. Installs dependencies (torch from the CPU index — no multi-GB CUDA build)
  3. Downloads the MediaPipe holistic landmarker model if missing
  4. Verifies the install and prints what's ready

Usage:
  python setup.py                # full setup
  python setup.py --check        # verify only, install nothing
  python setup.py --no-whisper   # skip torch/whisper (sign→speech only, much smaller)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "signslator", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")

# Core packages needed for sign→speech (camera, landmarks, TTS, classifier)
CORE_PACKAGES = [
    "opencv-python",
    "mediapipe",
    "numpy",
    "scipy",
    "pillow",
    "requests",
    "h5py",
    "pyttsx3",
]

# Extra packages for speech→sign (mic + Whisper ASR)
SPEECH_PACKAGES = [
    "sounddevice",
]

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw)


def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 9):
        print("ERROR: Python 3.9+ required.")
        return False
    return True


def already_installed(module_name):
    """Check importability without installing."""
    return subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def install_core():
    print("\n[1/4] Installing core packages (camera, landmarks, classifier)...")
    missing = []
    for pkg in CORE_PACKAGES:
        mod = pkg.replace("-", "_").replace("opencv_python", "cv2").replace("pyttsx3", "pyttsx3")
        if pkg == "opencv-python":
            mod = "cv2"
        if not already_installed(mod):
            missing.append(pkg)

    if not missing:
        print("  All core packages already installed.")
        return True

    print(f"  Installing: {', '.join(missing)}")
    r = run([sys.executable, "-m", "pip", "install"] + missing)
    return r.returncode == 0


def install_speech(with_whisper=True):
    print("\n[2/4] Installing speech packages...")

    missing = [p for p in SPEECH_PACKAGES if not already_installed(p)]
    if missing:
        print(f"  Installing: {', '.join(missing)}")
        run([sys.executable, "-m", "pip", "install"] + missing)

    if not with_whisper:
        print("  Skipping Whisper (--no-whisper): speech→sign will be unavailable.")
        return True

    if already_installed("whisper") and already_installed("torch"):
        print("  Whisper + torch already installed.")
        return True

    print("  Installing torch (CPU build — this is the big one, ~200 MB)...")
    r = run([sys.executable, "-m", "pip", "install",
             "torch", "torchvision", "--index-url", TORCH_INDEX])
    if r.returncode != 0:
        print("  WARNING: torch install failed. Retrying from PyPI...")
        run([sys.executable, "-m", "pip", "install", "torch", "torchvision"])

    if not already_installed("whisper"):
        print("  Installing openai-whisper...")
        run([sys.executable, "-m", "pip", "install", "openai-whisper"])

    return already_installed("whisper")


def download_model():
    print("\n[3/4] Checking MediaPipe model...")
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"  Already present: {MODEL_PATH} ({size_mb:.1f} MB)")
        return True

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"  Downloading holistic_landmarker.task (~14 MB)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"  ERROR downloading model: {e}")
        print(f"  Download manually and place at: {MODEL_PATH}")
        print(f"  URL: {MODEL_URL}")
        return False


def verify(with_whisper=True):
    print("\n[4/4] Verifying...")
    checks = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pyttsx3", "TTS (pyttsx3)"),
        ("h5py", "h5py"),
        ("PIL", "Pillow"),
    ]
    if with_whisper:
        checks += [("torch", "PyTorch"), ("whisper", "Whisper"), ("sounddevice", "sounddevice")]

    ok = True
    for mod, label in checks:
        good = already_installed(mod)
        print(f"  [{'OK ' if good else 'FAIL'}] {label}")
        ok = ok and good

    if not os.path.exists(MODEL_PATH):
        print("  [FAIL] MediaPipe model file")
        ok = False
    else:
        print("  [OK ] MediaPipe model file")

    # Check for signs
    signs_dir = os.path.join(HERE, "signslator", "data", "signs")
    n_signs = 0
    if os.path.isdir(signs_dir):
        n_signs = len([d for d in os.listdir(signs_dir)
                       if os.path.isdir(os.path.join(signs_dir, d))])
    print(f"  [{'OK ' if n_signs else '-- '}] Sign data: {n_signs} sign(s)")
    if n_signs == 0:
        print("        → Run:  python run.py isl-download")

    return ok


def main():
    ap = argparse.ArgumentParser(description="SignSlator setup")
    ap.add_argument("--check", action="store_true", help="Verify only, install nothing")
    ap.add_argument("--no-whisper", action="store_true",
                    help="Skip torch/Whisper (sign→speech only, much smaller download)")
    args = ap.parse_args()

    print("=" * 60)
    print("SignSlator setup")
    print("=" * 60)
    print()

    if not check_python():
        sys.exit(1)

    if args.check:
        ok = verify(with_whisper=not args.no_whisper)
        print()
        print("Ready to use." if ok else "Some pieces are missing — run: python setup.py")
        sys.exit(0 if ok else 1)

    with_whisper = not args.no_whisper

    if not install_core():
        print("\nERROR: core install failed.")
        sys.exit(1)

    install_speech(with_whisper=with_whisper)
    download_model()
    ok = verify(with_whisper=with_whisper)

    print()
    print("=" * 60)
    if ok:
        print("Setup complete. Next steps:")
        print("  python run.py isl-download    # fetch ISL signs")
        print("  python run.py recognize       # sign → speech")
        if with_whisper:
            print("  python run.py speech-to-sign  # speech → sign")
    else:
        print("Setup finished with warnings — see the FAIL lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
