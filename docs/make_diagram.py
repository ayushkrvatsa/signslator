"""
Draw the SignSlator architecture diagram as a PNG for the README / slides.

Pure Pillow — no native SVG libraries needed.

Usage:  python docs/make_diagram.py
"""

import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "architecture.png")

W, H = 1800, 840
BG = (14, 17, 23)
PANEL_BLUE = (32, 44, 66)
BORDER_BLUE = (58, 74, 102)
PANEL_PURPLE = (48, 36, 70)
BORDER_PURPLE = (74, 58, 102)
TEXT = (232, 236, 244)
MUTED = (139, 147, 167)
ARROW = (139, 147, 167)
OUT_BLUE = (31, 58, 42)
OUT_BLUE_T = (168, 230, 192)
OUT_PURPLE = (58, 42, 31)
OUT_PURPLE_T = (240, 208, 168)


def font(size, bold=False):
    names = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in names:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_TITLE = font(40, True)
F_SUB = font(21)
F_ROW = font(22, True)
F_BOX = font(21, True)
F_SMALL = font(16)
F_INFO_T = font(22, True)
F_INFO = font(18)
F_INFO_S = font(16)


def center_text(d, cx, y, text, f, fill):
    bb = d.textbbox((0, 0), text, font=f)
    d.text((cx - (bb[2] - bb[0]) / 2, y), text, font=f, fill=fill)


def box(d, x, y, w, h, fill, border, lines, out_fill=None, out_text=None):
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill, outline=border, width=2)
    cx = x + w / 2
    if len(lines) == 1:
        center_text(d, cx, y + h / 2 - 11, lines[0], F_BOX, TEXT)
    elif len(lines) == 2:
        center_text(d, cx, y + 14, lines[0], F_BOX, TEXT)
        center_text(d, cx, y + 44, lines[1], F_SMALL, MUTED)
    else:
        center_text(d, cx, y + 10, lines[0], F_BOX, TEXT)
        center_text(d, cx, y + 38, lines[1], F_BOX, TEXT)
        center_text(d, cx, y + 66, lines[2], F_SMALL, MUTED)


def arrow(d, x1, y, x2):
    d.line([(x1, y), (x2 - 8, y)], fill=ARROW, width=3)
    d.polygon([(x2 - 8, y - 6), (x2, y), (x2 - 8, y + 6)], fill=ARROW)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    center_text(d, W / 2, 28, "SignSlator — ISL Sign Language ↔ Speech", F_TITLE, TEXT)
    center_text(d, W / 2, 82, "Bidirectional pipeline  ·  all free and open-source", F_SUB, MUTED)

    # ── Row 1: SIGN → SPEECH ──
    d.text((60, 132), "SIGN → SPEECH", font=F_ROW, fill=(127, 178, 255))

    y1, h = 166, 104
    boxes1 = [
        (60, 216, ["Webcam", "OpenCV capture"]),
        (320, 258, ["MediaPipe Hand", "21 landmarks / hand"]),
        (600, 240, ["Hand landmarks", "21 × 2 · normalized"]),
        (880, 240, ["DTW classifier", "auto-calibrated"]),
        (1160, 216, ["pyttsx3", "offline TTS"]),
    ]
    for x, w, lines in boxes1:
        box(d, x, y1, w, h, PANEL_BLUE, BORDER_BLUE, lines)

    # output box
    d.rounded_rectangle([1440, y1, 1740, y1 + h], radius=14, fill=OUT_BLUE,
                        outline=(47, 107, 69), width=2)
    center_text(d, 1590, y1 + 26, "Spoken", F_BOX, OUT_BLUE_T)
    center_text(d, 1590, y1 + 56, "word", F_BOX, OUT_BLUE_T)

    for x1, x2 in [(276, 316), (578, 596), (840, 876), (1120, 1156), (1376, 1436)]:
        arrow(d, x1, y1 + h / 2, x2)

    # ── Row 2: SPEECH → SIGN ──
    d.text((60, 330), "SPEECH → SIGN", font=F_ROW, fill=(217, 168, 255))

    y2 = 364
    boxes2 = [
        (60, 216, ["Microphone", "16 kHz capture"]),
        (320, 258, ["Whisper ASR", "offline, local, CPU"]),
        (600, 240, ["Text", "word lookup"]),
        (880, 240, ["Sign templates", "your recordings"]),
        (1160, 216, ["Landmark replay", ""]),
    ]
    for x, w, lines in boxes2:
        box(d, x, y2, w, h, PANEL_PURPLE, BORDER_PURPLE, lines)

    d.rounded_rectangle([1440, y2, 1740, y2 + h], radius=14, fill=OUT_PURPLE,
                        outline=(107, 74, 47), width=2)
    center_text(d, 1590, y2 + 26, "Animated", F_BOX, OUT_PURPLE_T)
    center_text(d, 1590, y2 + 56, "sign", F_BOX, OUT_PURPLE_T)

    for x1, x2 in [(276, 316), (578, 596), (840, 876), (1120, 1156), (1376, 1436)]:
        arrow(d, x1, y2 + h / 2, x2)

    # ── Bottom info panels ──
    y3, h3 = 530, 250
    d.rounded_rectangle([60, y3, 870, y3 + h3], radius=14, fill=(20, 26, 38),
                        outline=(42, 52, 72), width=2)
    d.text((84, y3 + 26), "Sign data — your own recordings", font=F_INFO_T, fill=TEXT)
    d.text((84, y3 + 70), "Templates are recordings of your hands", font=F_INFO, fill=(168, 176, 192))
    d.text((84, y3 + 100), "5+ samples per sign, stored locally", font=F_INFO, fill=(168, 176, 192))
    d.text((84, y3 + 130), "Optional: ISL500 public dataset instead", font=F_INFO, fill=(168, 176, 192))
    d.text((84, y3 + 180), "No sign data ships with this repository", font=F_INFO_S, fill=(111, 122, 143))
    d.text((84, y3 + 205), "nothing leaves your machine", font=F_INFO_S, fill=(111, 122, 143))

    d.rounded_rectangle([930, y3, 1740, y3 + h3], radius=14, fill=(20, 26, 38),
                        outline=(42, 52, 72), width=2)
    d.text((954, y3 + 26), "Verification", font=F_INFO_T, fill=TEXT)
    d.text((954, y3 + 70), "Auto-calibrated threshold from your own templates", font=F_INFO, fill=(168, 176, 192))
    d.text((954, y3 + 100), "Runner-up margin rejects ambiguous matches", font=F_INFO, fill=(168, 176, 192))
    d.text((954, y3 + 130), "Scale normalization removes hand-size bias", font=F_INFO, fill=(168, 176, 192))
    d.text((954, y3 + 180), "Measure it yourself: python run.py self-test", font=F_INFO_S, fill=(111, 122, 143))
    d.text((954, y3 + 205), "no accuracy is claimed without your data", font=F_INFO_S, fill=(111, 122, 143))

    img.save(OUT, "PNG", optimize=True)
    print(f"Saved {OUT}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
