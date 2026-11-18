"""
Sign reference guide — see what each sign actually looks like.

Two sources:
  1. Landmark animation (instant, from templates you already have)
     Renders each sign's hand motion to an animated GIF.
  2. Real reference video (download on demand, ~21 MB each)
     The actual ISL500 signer performing the sign.

Usage:
  python sign_guide.py                      # render GIFs for all trained signs
  python sign_guide.py --sign HELLO         # just one
  python sign_guide.py --video HELLO        # download the real video too
  python sign_guide.py --list               # list signs + what's available
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator import get_vocabulary
from signslator.speech_to_sign import load_sign_templates, HAND_CONN

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
GUIDE_DIR = os.path.join(HERE, "sign_guide")
HF_BASE = "https://huggingface.co/datasets/ISL500/ISL-DATA/resolve/main"

# Canvas
W, H = 480, 360

# Hand landmark connections for drawing
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def render_frame(features: np.ndarray, label: str, frame_i: int, total: int):
    """Render one frame of a sign's landmark animation to a PIL image."""
    img = Image.new("RGB", (W, H), (18, 18, 24))
    draw = ImageDraw.Draw(img)

    def hand_points(offset):
        """Extract 21 (x,y) points for one hand, or [] if absent."""
        pts = []
        for i in range(21):
            x = features[offset + i * 3]
            y = features[offset + i * 3 + 1]
            pts.append((x, y))
        present = any(x != 0 or y != 0 for x, y in pts)
        return pts if present else []

    left = hand_points(0)
    right = hand_points(63)

    # Fit both hands into the canvas with shared scaling so relative
    # hand position is preserved (important for two-handed signs).
    all_pts = left + right
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        margin = 40
        scale = min((W - 2 * margin) / span_x, (H - 2 * margin) / span_y)

        def to_px(p):
            px = margin + (p[0] - min_x) * scale
            py = margin + (p[1] - min_y) * scale
            return (px, py)

        for pts, color in ((left, (90, 170, 255)), (right, (255, 120, 120))):
            if not pts:
                continue
            px = [to_px(p) for p in pts]
            for i, j in CONNECTIONS:
                if i < len(px) and j < len(px):
                    draw.line([px[i], px[j]], fill=color, width=3)
            for p in px:
                draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=color)

    # Label + progress
    draw.text((12, 10), label.upper(), fill=(255, 255, 255))
    draw.text((12, H - 20), f"frame {frame_i+1}/{total}", fill=(150, 150, 150))
    bar_w = int((W - 24) * (frame_i + 1) / total)
    draw.rectangle([12, H - 6, 12 + bar_w, H - 3], fill=(80, 200, 120))

    return img


def render_gif(sign: str, max_frames: int = 40) -> str:
    """Render a sign's animation to an animated GIF. Returns the path."""
    templates = load_sign_templates(sign)
    if not templates:
        print(f"  {sign}: no templates — download it first")
        return None

    seq = templates[0]
    n = seq.shape[0]

    # Sample evenly down to max_frames
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).astype(int)
        seq = seq[idx]
        n = max_frames

    frames = [render_frame(seq[i], sign, i, n) for i in range(n)]

    os.makedirs(GUIDE_DIR, exist_ok=True)
    out = os.path.join(GUIDE_DIR, f"{sign.upper()}.gif")
    # Hold the final frame a moment by duplicating it
    frames = frames + [frames[-1]] * 4
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print(f"  {sign}: {out}")
    return out


def download_reference_video(sign: str, user: str = "User001") -> str:
    """Download the real ISL500 video for a sign. Returns the path."""
    import requests
    import re

    # Find the video filename for this sign from the mapping CSV
    csv_url = f"{HF_BASE}/MappingFiles/PersonDependentTrain.csv"
    r = requests.get(csv_url, timeout=60)
    r.raise_for_status()

    user_dir = "ISL_DATA_USER" + user.replace("User", "").zfill(3)
    pattern = re.compile(rf"{sign}__session\d+__clip\d+\.h5$", re.IGNORECASE)

    filename = None
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("fold"):
            continue
        base = line.split("/")[-1]
        if user_dir in line and pattern.search(base):
            filename = base.replace(".h5", ".mp4")
            break

    if not filename:
        print(f"  {sign}: no reference video found")
        return None

    os.makedirs(GUIDE_DIR, exist_ok=True)
    out = os.path.join(GUIDE_DIR, f"{sign.upper()}_reference.mp4")
    if os.path.exists(out):
        print(f"  {sign}: already have {out}")
        return out

    url = f"{HF_BASE}/Videos/{user}/{filename}"
    print(f"  {sign}: downloading {filename} (~21 MB)...")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    print(f"  {sign}: saved {out}")
    return out


def list_signs():
    signs = get_vocabulary()
    if not signs:
        print("No signs downloaded. Run: python run.py isl-download")
        return
    print(f"\n{len(signs)} signs available:\n")
    for s in signs:
        t = load_sign_templates(s)
        gif = os.path.join(GUIDE_DIR, f"{s}.gif")
        vid = os.path.join(GUIDE_DIR, f"{s}_reference.mp4")
        marks = []
        if os.path.exists(gif):
            marks.append("GIF")
        if os.path.exists(vid):
            marks.append("VIDEO")
        tag = f"  [{', '.join(marks)}]" if marks else ""
        print(f"  {s:<10} {len(t)} template(s){tag}")
    print(f"\nGuide folder: {GUIDE_DIR}")


def main():
    ap = argparse.ArgumentParser(description="Sign reference guide")
    ap.add_argument("--sign", help="Render one sign only")
    ap.add_argument("--video", metavar="SIGN",
                    help="Download the real reference video for a sign")
    ap.add_argument("--user", default="User001", help="ISL500 signer for videos")
    ap.add_argument("--list", action="store_true", help="List signs and assets")
    args = ap.parse_args()

    if args.list:
        list_signs()
        return

    if args.video:
        download_reference_video(args.video.upper(), args.user)
        return

    if not PIL_OK:
        print("Pillow not installed. Run: python -m pip install pillow")
        return

    signs = [args.sign.upper()] if args.sign else get_vocabulary()
    if not signs:
        print("No signs available. Run: python run.py isl-download")
        return

    print(f"Rendering {len(signs)} sign animations to GIF...\n")
    for s in signs:
        render_gif(s)
    print(f"\nOpen the GIFs in: {GUIDE_DIR}")


if __name__ == "__main__":
    main()
