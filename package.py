#!/usr/bin/env python3
"""
Package SignSlator for transfer to another machine.

Produces: signslator-portable.zip containing
  - all source code
  - the MediaPipe hand model (8 MB)
  - your trained sign templates
  - setup.py + requirements.txt + README.md

Not included (re-created automatically on the target machine):
  - Whisper model  (~140 MB, downloads on first speech-to-sign run)
  - the old holistic model (dev-only, used by compare_detectors.py)
  - isl500_cache   (raw .h5 files, only needed for re-downloading signs)

Usage:  python package.py
"""

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "signslator-portable.zip")

# (source path relative to HERE, is_directory)
INCLUDE = [
    ("run.py", False),
    ("setup.py", False),
    ("package.py", False),
    ("requirements.txt", False),
    ("README.md", False),
    ("LICENSE", False),
    (".gitignore", False),
    (".gitattributes", False),
    ("docs", True),
    ("signslator/__init__.py", False),
    ("signslator/demo_camera.py", False),
    ("signslator/demo_speech.py", False),
    ("signslator/sign_to_speech.py", False),
    ("signslator/speech_to_sign.py", False),
    ("signslator/isl500_loader.py", False),
    ("signslator/sign_guide.py", False),
    ("signslator/sign_chart.py", False),
    ("signslator/self_test.py", False),
    ("signslator/full_eval.py", False),
    ("signslator/test_isl500_crossval.py", False),
    ("signslator/test_isl500_classifier.py", False),
    ("signslator/test_isl500_recognize.py", False),
    ("signslator/test_normalization.py", False),
    ("signslator/test_hand_features.py", False),
    ("signslator/test_roundtrip.py", False),
    ("signslator/test_matcher_verify.py", False),
    ("signslator/probe_hand.py", False),
    ("signslator/compare_detectors.py", False),
    ("signslator/models", True),
    ("signslator/data/signs", True),
    ("signslator/sign_guide", True),
]

# Never ship these
EXCLUDE_EXT = {".h5", ".mp4", ".pyc"}
EXCLUDE_DIRS = {"__pycache__", "isl500_cache", "captures", ".git"}
# Dev-only: the old model, kept locally for compare_detectors.py.
EXCLUDE_FILES = {"holistic_landmarker.task"}


def should_skip(path):
    parts = set(path.replace("\\", "/").split("/"))
    if parts & EXCLUDE_DIRS:
        return True
    if os.path.basename(path) in EXCLUDE_FILES:
        return True
    return os.path.splitext(path)[1] in EXCLUDE_EXT


def main():
    ap = argparse.ArgumentParser(description="Package SignSlator for transfer")
    ap.add_argument("--no-data", action="store_true",
                    help="Exclude sign templates — recipient trains their own signs")
    ap.add_argument("--no-guide", action="store_true",
                    help="Exclude generated GIFs and chart")
    ap.add_argument("--out", default=None, help="Output zip path")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = args.out

    include = list(INCLUDE)
    if args.no_data:
        include = [(p, d) for p, d in include if p != "signslator/data/signs"]
    if args.no_guide:
        include = [(p, d) for p, d in include if p != "signslator/sign_guide"]

    if os.path.exists(OUT):
        os.remove(OUT)

    count = 0
    total_bytes = 0

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, is_dir in include:
            src = os.path.join(HERE, rel)
            if not os.path.exists(src):
                print(f"  skip (missing): {rel}")
                continue

            if is_dir:
                for root, dirs, files in os.walk(src):
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    for f in files:
                        full = os.path.join(root, f)
                        arc = os.path.relpath(full, HERE).replace("\\", "/")
                        if should_skip(arc):
                            continue
                        z.write(full, arc)
                        count += 1
                        total_bytes += os.path.getsize(full)
            else:
                arc = rel.replace("\\", "/")
                if should_skip(arc):
                    continue
                z.write(src, arc)
                count += 1
                total_bytes += os.path.getsize(src)

    if count == 0:
        print("ERROR: nothing was packaged.")
        return 1

    size = os.path.getsize(OUT)
    print(f"\nPackaged {count} files -> {OUT}")
    print(f"Zip size:   {size/1024/1024:.1f} MB")
    print(f"Unpacked:   {total_bytes/1024/1024:.1f} MB")
    print()
    print("Contents:")
    with zipfile.ZipFile(OUT) as z:
        for n in sorted(z.namelist()):
            print(f"  {n}")
    print()
    print("On the target machine:")
    print("  1. Unzip the folder anywhere")
    print("  2. cd into it")
    print("  3. python setup.py        # installs dependencies + model")
    if args.no_data:
        print("  4. python run.py train HELLO    # record your first sign")
    print("  5. python run.py recognize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
