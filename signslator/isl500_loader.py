"""
ISL500 Dataset Loader — downloads ISL landmarks and converts to SignSlator template format.

Downloads MediaPipe landmark files (.h5) from HuggingFace ISL500 dataset
and converts them to our data/signs/<WORD>/template_<i>.npy format.

Usage:
  python isl500_loader.py              # Download USER001, all 500 words (2 sessions each)
  python isl500_loader.py --user USER003   # Different user
  python isl500_loader.py --words HELLO YES NO   # Only specific words
  python isl500_loader.py --sessions 1        # Only 1 session per word (faster)

Output: data/signs/<WORD>/template_<session>.npy  (numpy arrays, same format as our trainer)
"""

import os
import sys
import argparse
import numpy as np
import h5py
import requests
import time
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signslator.sign_to_speech import _normalize_hand, NORMALIZE_SCALE

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "signslator", "data", "signs")
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "isl500_cache")

HUGGINGFACE_BASE = "https://huggingface.co/datasets/ISL500/ISL-DATA/resolve/main"

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "isl500_config.json")

# ── Helpers ─────────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Download a file with progress feedback."""
    if os.path.exists(dest_path):
        return True  # Already downloaded

    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            if total:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 1024 * 1024:
                        pct = downloaded / total * 100
                        print(f"    {os.path.basename(dest_path)}: {pct:.0f}%", end='\r')
            else:
                f.write(r.content)
        print(f"    Saved: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def parse_word_from_filename(filename: str) -> str:
    """Extract sign word from .h5 filename.
    Pattern: <WORD>__session<N>__clip<N>.h5
    """
    match = re.search(r'^([A-Za-z]+)__session\d+__clip\d+\.h5$', filename)
    if match:
        return match.group(1)
    # Handle prefixed filenames
    match = re.search(r'_([^_]+)__session\d+__clip\d+\.h5$', filename)
    if match:
        return match.group(1)
    return None


def ist500_to_features(h5_data: np.ndarray, normalize: bool = None) -> np.ndarray:
    """
    Convert ISL500 landmark data (frames, 2 hands, 21 landmarks, 3 coords)
    to our feature format (frames, 144).

    Our feature format:
      [0:63]   = left hand landmarks (21 × 3 = 63)
      [63:126] = right hand landmarks (21 × 3 = 63)
      [126:144] = derived features (hand centers, distance) = 18
    """
    if normalize is None:
        normalize = NORMALIZE_SCALE
    norm_fn = _normalize_hand if normalize else (lambda p: p)

    n_frames = h5_data.shape[0]
    features = np.zeros((n_frames, 144), dtype=np.float32)

    for f in range(n_frames):
        left = h5_data[f, 0]   # (21, 3)
        right = h5_data[f, 1]  # (21, 3)

        # Same scale/translation normalization the live pipeline uses,
        # so dataset templates and your webcam features are comparable.
        left_n = norm_fn(left)
        right_n = norm_fn(right)

        left_present = np.any(left != 0)
        right_present = np.any(right != 0)

        features[f, 0:63] = left_n.flatten()
        features[f, 63:126] = right_n.flatten()

        # Derived features (computed in normalized space)
        if left_present and right_present:
            left_center = left_n.mean(axis=0)
            right_center = right_n.mean(axis=0)
            features[f, 126] = np.linalg.norm(left_center - right_center)
            features[f, 127] = left_center[0]
            features[f, 128] = left_center[1]
            features[f, 129] = right_center[0]
            features[f, 130] = right_center[1]
        elif left_present:
            c = left_n.mean(axis=0)
            features[f, 127] = c[0]
            features[f, 128] = c[1]
        elif right_present:
            c = right_n.mean(axis=0)
            features[f, 129] = c[0]
            features[f, 130] = c[1]

    return features


def load_mapping_csv(url: str) -> dict:
    """
    Download and parse the mapping CSV.
    Returns: {word: [list of (session, clip, full_path, filename) tuples]}
    """
    print(f"Downloading mapping CSV: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    content = r.text

    word_map = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('fold'):
            continue

        full_path = line
        filename = line.split('/')[-1]
        word = parse_word_from_filename(filename)
        if not word:
            continue

        # Extract session and clip from filename
        session_match = re.search(r'session(\d+)', filename)
        clip_match = re.search(r'clip(\d+)', filename)
        session = int(session_match.group(1)) if session_match else 0
        clip = int(clip_match.group(1)) if clip_match else 0

        if word not in word_map:
            word_map[word] = []
        word_map[word].append((session, clip, full_path, filename))

    return word_map


def download_and_convert(user: str, words: list = None, max_sessions: int = 2):
    """
    Download ISL500 landmarks for a specific user and convert to templates.
    """
    print(f"\n{'='*60}")
    print(f"ISL500 Loader — User: {user}")
    print(f"{'='*60}\n")

    ensure_dirs()

    # Step 1: Download mapping CSVs and build file list
    print("Step 1: Building file list from mapping CSVs...")
    train_url = f"{HUGGINGFACE_BASE}/MappingFiles/PersonDependentTrain.csv"
    test_url = f"{HUGGINGFACE_BASE}/MappingFiles/PersonDependentTest.csv"

    word_map = {}
    for url, label in [(train_url, "train"), (test_url, "test")]:
        try:
            csv_map = load_mapping_csv(url)
            for word, entries in csv_map.items():
                if words is None or word in words or word.upper() in [w.upper() for w in words]:
                    # Filter entries for this user by checking path
                    user_entries = []
                    for session, clip, full_path, filename in entries:
                        # full_path contains user directory like .../ISL_DATA_USER001/...
                        if user in full_path:
                            user_entries.append((session, clip, filename))
                    if user_entries:
                        if word not in word_map:
                            word_map[word] = []
                        word_map[word].extend(user_entries)
        except Exception as e:
            print(f"  Warning: Could not load {label} CSV: {e}")

    if not word_map:
        print(f"ERROR: No data found for user {user}")
        return False

    print(f"  Found {len(word_map)} signs for {user}")
    if words:
        print(f"  Filtering to requested words: {len([w for w in words if w in word_map])}")

    # Step 2: Download .h5 files and convert
    print(f"\nStep 2: Downloading landmarks and converting ({len(word_map)} signs)...")
    print(f"  Estimated download: ~{len(word_map) * max_sessions * 30 / 1024:.0f} MB\n")

    success_count = 0
    fail_count = 0

    for word in sorted(word_map.keys()):
        if words and word not in words:
            continue

        sign_dir = os.path.join(DATA_DIR, word.upper())
        os.makedirs(sign_dir, exist_ok=True)

        entries = word_map[word]
        # Sort by session, limit to max_sessions
        entries.sort(key=lambda x: x[0])
        entries = entries[:max_sessions]

        word_success = 0
        for session, clip, filename in entries:
            # Download .h5
            h5_url = f"{HUGGINGFACE_BASE}/Landmarks/MediaPipe/{user}/{filename}"
            cache_path = os.path.join(DOWNLOAD_DIR, filename)

            if not download_file(h5_url, cache_path):
                fail_count += 1
                continue

            # Load and convert
            try:
                with h5py.File(cache_path, 'r') as f:
                    h5_data = f['intermediate'][:]

                features = ist500_to_features(h5_data)

                # Save as template
                template_name = f"template_{session}"
                template_path = os.path.join(sign_dir, f"{template_name}.npy")
                np.save(template_path, features)

                word_success += 1
                success_count += 1

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                fail_count += 1
                continue

        if word_success > 0:
            print(f"  ✓ {word}: {word_success} template(s) saved")
        else:
            print(f"  ✗ {word}: failed")

    print(f"\n{'='*60}")
    print(f"DONE: {success_count} templates saved, {fail_count} failed")
    print(f"Signs with templates: {len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])}")
    print(f"{'='*60}")

    return True


def list_downloaded_signs():
    """List signs that have been downloaded."""
    if not os.path.exists(DATA_DIR):
        print("No signs downloaded yet.")
        return
    print("\nDownloaded ISL signs:")
    signs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    for sign in signs:
        templates = [f for f in os.listdir(os.path.join(DATA_DIR, sign)) if f.endswith('.npy')]
        print(f"  {sign}: {len(templates)} templates")
    print(f"\nTotal: {len(signs)} signs, {sum(len([f for f in os.listdir(os.path.join(DATA_DIR, d)) if f.endswith('.npy')]) for d in signs)} templates")


def reconvert_cached():
    """
    Re-process cached .h5 files into templates WITHOUT re-downloading.
    Needed when the feature format changes (e.g. scale normalization).
    """
    if not os.path.isdir(DOWNLOAD_DIR):
        print("No cache directory — run a download first.")
        return
    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".h5")]
    if not files:
        print("No cached .h5 files — run a download first.")
        return

    print(f"Re-converting {len(files)} cached files with the current feature format...\n")
    ok = fail = 0
    for fname in sorted(files):
        word = parse_word_from_filename(fname)
        session_m = re.search(r"session(\d+)", fname)
        session = int(session_m.group(1)) if session_m else 0
        if not word:
            continue
        try:
            with h5py.File(os.path.join(DOWNLOAD_DIR, fname), "r") as f:
                h5 = f["intermediate"][:]
            feats = ist500_to_features(h5)
            sign_dir = os.path.join(DATA_DIR, word.upper())
            os.makedirs(sign_dir, exist_ok=True)
            np.save(os.path.join(sign_dir, f"template_{session}.npy"), feats)
            ok += 1
        except Exception as e:
            print(f"  FAILED {fname}: {e}")
            fail += 1

    print(f"\nRe-converted {ok} templates, {fail} failed.")
    list_downloaded_signs()


def main():
    parser = argparse.ArgumentParser(description="ISL500 Dataset Loader")
    parser.add_argument("--user", default="ISL_DATA_USER001",
                       help="User to download (default: ISL_DATA_USER001)")
    parser.add_argument("--words", nargs="*", default=None,
                       help="Specific words to download (default: all 500)")
    parser.add_argument("--sessions", type=int, default=2,
                       help="Max sessions per word (default: 2)")
    parser.add_argument("--list", action="store_true",
                       help="List downloaded signs")
    parser.add_argument("--clear", action="store_true",
                       help="Clear all downloaded signs and cache")
    parser.add_argument("--reconvert", action="store_true",
                       help="Re-convert cached .h5 files (no re-download)")

    args = parser.parse_args()

    if args.clear:
        import shutil
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        if os.path.exists(DOWNLOAD_DIR):
            shutil.rmtree(DOWNLOAD_DIR)
        print("Cleared data/signs/ and isl500_cache/")
        return

    if args.list:
        list_downloaded_signs()
        return

    if args.reconvert:
        reconvert_cached()
        return

    # Validate words against known ISL500 vocabulary
    if args.words:
        # Quick check — try to load mapping to verify words exist
        try:
            train_url = f"{HUGGINGFACE_BASE}/MappingFiles/PersonDependentTrain.csv"
            word_map = load_mapping_csv(train_url)
            known_words = set(word_map.keys())
            unknown = [w for w in args.words if w.upper() not in known_words and w not in known_words]
            if unknown:
                print(f"Warning: These words may not exist in ISL500: {unknown}")
                print(f"Available words (first 30): {sorted(list(known_words))[:30]}")
        except Exception as e:
            print(f"Could not verify words (network issue): {e}")

    download_and_convert(args.user, args.words, args.sessions)
    list_downloaded_signs()


if __name__ == "__main__":
    main()
