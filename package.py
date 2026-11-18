#!/usr/bin/env python3
"""
Package SignSlator for transfer to another machine.

Produces: signslator-portable.zip containing
  - all source code
  - the MediaPipe model (14 MB)
  - your downloaded sign templates
  - sign guide GIFs
  - setup.py + requirements.txt + README.md

Not included (re-created automatically on the target machine):
  - Whisper model  (~140 MB, downloads on first speech-to-sign run)
  - isl500_cache   (raw .h5 files, only needed for re-downloading signs)

Usage:  python package.py
"""

import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "signslator-portable.zip")

# (source path relative to HERE, is_directory)
INCLUDE = [
    ("run.py", False),
    ("setup.py", False),
    ("requirements.txt", False),
    ("README.md", False),
    ("signslator/__init__.py", False),
    ("signslator/demo_camera.py", False),
    ("signslator/demo_speech.py", False),
    ("signslator/sign_to_speech.py", False),
    ("signslator/speech_to_sign.py", False),
    ("signslator/isl500_loader.py", False),
    ("signslator/sign_guide.py", False),
    ("signslator/sign_chart.py", False),
    ("signslator/self_test.py", False),
    ("signslator/test_isl500_crossval.py", False),
    ("signslator/test_isl500_classifier.py", False),
    ("signslator/test_isl500_recognize.py", False),
    ("signslator/test_normalization.py", False),
    ("signslator/models", True),
    ("signslator/data/signs", True),
    ("signslator/sign_guide", True),
]

# Never ship these
EXCLUDE_EXT = {".h5", ".mp4", ".pyc"}
EXCLUDE_DIRS = {"__pycache__", "isl500_cache", "captures"}


def should_skip(path):
    parts = set(path.replace("\\", "/").split("/"))
    if parts & EXCLUDE_DIRS:
        return True
    return os.path.splitext(path)[1] in EXCLUDE_EXT


def main():
    if os.path.exists(OUT):
        os.remove(OUT)

    count = 0
    total_bytes = 0

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, is_dir in INCLUDE:
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
    print(f"\nPackaged {count} files → {OUT}")
    print(f"Zip size:   {size/1024/1024:.1f} MB")
    print(f"Unpacked:   {total_bytes/1024/1024:.1f} MB")
    print()
    print("Contents:")
    with zipfile.ZipFile(OUT) as z:
        for n in sorted(z.namelist()):
            print(f"  {n}")
    print()
    print("On the other machine:")
    print("  1. Unzip the folder anywhere")
    print("  2. cd into it")
    print("  3. python setup.py        # installs dependencies + model")
    print("  4. python run.py recognize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
