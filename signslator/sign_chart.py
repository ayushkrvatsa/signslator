"""
Sign chart generator — a visual reference poster of all your signs.

For each sign it draws the hand skeleton at three moments (start / middle / end
of the movement), so you can see both the shape and the motion, then tiles them
into a poster with the word underneath.

Outputs:
  signslator/sign_guide/sign_chart.png   (large poster image)
  signslator/sign_guide/sign_chart.html  (crisp, zoomable, prints nicely)

Usage:
  python run.py chart              # all downloaded signs
  python run.py chart --sign HELLO # one sign
  python run.py chart --cols 4     # poster layout
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator import get_vocabulary
from signslator.speech_to_sign import load_sign_templates

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

HERE = os.path.dirname(os.path.abspath(__file__))
GUIDE_DIR = os.path.join(HERE, "sign_guide")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _font(size):
    """Try to load a real TTF; fall back to PIL's default bitmap font."""
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def hand_offsets(features):
    """Return (left_pts, right_pts) as (21,2) arrays, or None when absent."""
    out = []
    for base in (0, 63):
        pts = features[base:base + 63].reshape(21, 3)[:, :2]
        out.append(pts if np.any(pts) else None)
    return out


def draw_pose(draw, features, box, pad=8):
    """Draw a hand skeleton inside box=(x0,y0,x1,y1). Returns True if drawn."""
    left, right = hand_offsets(features)
    all_pts = [p for p in (left, right) if p is not None]
    if not all_pts:
        return False

    flat = np.vstack(all_pts)
    min_xy = flat.min(axis=0)
    max_xy = flat.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    x0, y0, x1, y1 = box
    avail_w = (x1 - x0) - 2 * pad
    avail_h = (y1 - y0) - 2 * pad
    scale = min(avail_w / span[0], avail_h / span[1])

    def to_px(p):
        return (
            x0 + pad + (p[0] - min_xy[0]) * scale,
            y0 + pad + (p[1] - min_xy[1]) * scale,
        )

    for pts, color in ((left, (95, 175, 255)), (right, (255, 120, 120))):
        if pts is None:
            continue
        px = [to_px(p) for p in pts]
        for i, j in CONNECTIONS:
            draw.line([px[i], px[j]], fill=color, width=2)
        for p in px:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
    return True


def pick_frames(seq, n=3):
    """Choose n frames spread across the sequence, skipping empty ones."""
    present = [i for i in range(seq.shape[0])
               if np.any(seq[i, 0:63]) or np.any(seq[i, 63:126])]
    if not present:
        return []
    if len(present) <= n:
        return present
    idx = np.linspace(0, len(present) - 1, n).round().astype(int)
    return [present[i] for i in idx]


def render_panel(sign, seq, frames, frame_w, frame_h, label_h):
    """Render one sign as a labelled strip of frames. Returns a PIL image."""
    panel = Image.new("RGB", (frame_w * 3, frame_h + label_h), (24, 24, 30))
    d = ImageDraw.Draw(panel)

    for k, fi in enumerate(frames[:3]):
        x0 = k * frame_w
        draw_pose(d, seq[fi], (x0, 0, x0 + frame_w, frame_h), pad=10)
        # frame separator
        if k:
            d.line([(x0, 0), (x0, frame_h)], fill=(60, 60, 70), width=1)

    # label
    f = _font(20)
    text = sign.upper()
    try:
        bbox = d.textbbox((0, 0), text, font=f)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = len(text) * 10
    d.text(((frame_w * 3 - tw) / 2, frame_h + 8), text, fill=(240, 240, 240), font=f)
    return panel


def build_chart(signs, cols=4, frame_w=180, frame_h=135, label_h=34):
    """Tile all sign panels into one poster image."""
    panels = []
    for s in signs:
        t = load_sign_templates(s)
        if not t:
            continue
        seq = t[0]
        frames = pick_frames(seq, 3)
        if not frames:
            continue
        panels.append((s, render_panel(s, seq, frames, frame_w, frame_h, label_h)))

    if not panels:
        print("No renderable signs.")
        return None, []

    panel_w = frame_w * 3
    panel_h = frame_h + label_h
    pad = 14
    header_h = 64

    rows = (len(panels) + cols - 1) // cols
    W = cols * panel_w + (cols + 1) * pad
    H = header_h + rows * panel_h + (rows + 1) * pad

    img = Image.new("RGB", (W, H), (16, 16, 22))
    d = ImageDraw.Draw(img)

    title_f = _font(30)
    sub_f = _font(15)
    d.text((pad, 12), "SignSlator — ISL Sign Chart", fill=(245, 245, 245), font=title_f)
    d.text((pad, 44), f"{len(panels)} signs · blue = one hand, red = the other · "
                      f"three moments of each movement",
           fill=(150, 150, 160), font=sub_f)

    for i, (sign, panel) in enumerate(panels):
        r, c = divmod(i, cols)
        x = pad + c * (panel_w + pad)
        y = header_h + pad + r * (panel_h + pad)
        img.paste(panel, (x, y))

    return img, [s for s, _ in panels]


def write_html(signs, out_path):
    """Write a zoomable HTML chart (crisper than PNG, prints well)."""
    rows = []
    for s in signs:
        gif = os.path.join(GUIDE_DIR, f"{s}.gif")
        if os.path.exists(gif):
            rows.append(
                f'<div class="card">'
                f'<img src="sign_guide/{s}.gif" alt="{s}">'
                f'<div class="label">{s.upper()}</div></div>'
            )
        else:
            rows.append(
                f'<div class="card empty"><div class="label">{s.upper()}</div>'
                f'<div class="note">no animation</div></div>'
            )

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SignSlator — ISL Sign Chart</title>
<style>
  body { background:#101016; color:#eaeaf0; font-family: system-ui, sans-serif;
         margin: 24px; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  .sub { color:#9a9aac; font-size: 14px; margin-bottom: 22px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(300px,1fr));
          gap: 16px; }
  .card { background:#18181f; border:1px solid #2a2a35; border-radius:10px;
          padding: 10px; text-align:center; }
  .card img { width:100%; border-radius:6px; display:block; }
  .label { font-size: 18px; font-weight: 600; margin-top: 10px; letter-spacing:.5px; }
  .note { color:#777; font-size: 12px; margin-top: 8px; }
  @media print { body { background:#fff; color:#000; } .card { border:1px solid #ccc; } }
</style></head><body>
<h1>SignSlator — ISL Sign Chart</h1>
<div class="sub">__N__ signs · each animation shows the hand movement for that word</div>
<div class="grid">
__ROWS__
</div></body></html>"""
    html = html.replace("__N__", str(len(signs))).replace("__ROWS__", "\n".join(rows))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description="Generate a sign chart")
    ap.add_argument("--sign", action="append", default=None,
                    help="Only these signs (repeatable)")
    ap.add_argument("--cols", type=int, default=4, help="Poster columns (default 4)")
    ap.add_argument("--html-only", action="store_true", help="Only write HTML")
    args = ap.parse_args()

    signs = [s.upper() for s in args.sign] if args.sign else get_vocabulary()
    if not signs:
        print("No signs. Run: python run.py isl-download")
        return

    os.makedirs(GUIDE_DIR, exist_ok=True)

    # Make sure GIFs exist for the HTML version
    if not args.html_only:
        try:
            from signslator.sign_guide import render_gif
            missing = [s for s in signs
                       if not os.path.exists(os.path.join(GUIDE_DIR, f"{s}.gif"))]
            if missing:
                print(f"Rendering {len(missing)} missing GIFs...")
                for s in missing:
                    render_gif(s)
        except Exception as e:
            print(f"(GIF step skipped: {e})")

    if not args.html_only:
        if not PIL_OK:
            print("Pillow not installed. Run: python -m pip install pillow")
            return
        print(f"Building chart for {len(signs)} signs...")
        img, rendered = build_chart(signs, cols=args.cols)
        if img is not None:
            out = os.path.join(GUIDE_DIR, "sign_chart.png")
            img.save(out, "PNG", optimize=True)
            print(f"  Saved: {out}  ({img.size[0]}x{img.size[1]})")

    html_out = os.path.join(GUIDE_DIR, "sign_chart.html")
    write_html(signs, html_out)
    print(f"  Saved: {html_out}")


if __name__ == "__main__":
    main()
