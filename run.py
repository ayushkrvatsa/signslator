#!/usr/bin/env python3
"""
SignSlator — Master Runner

Unified entry point for the bidirectional sign language system.
Usage:
  python run.py camera          # Phase 1A: camera + landmarks
  python run.py speech          # Phase 1B: speech→text
  python run.py train <sign>    # Phase 2: train a sign
  python run.py recognize       # Phase 2: live sign recognition
  python run.py speech-to-sign  # Phase 3: speech→sign animation
  python run.py list            # List trained signs
  python run.py demo            # Run all phases sequentially (quick demo)
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import subprocess


def run_camera():
    from signslator.demo_camera import main as camera_main
    camera_main()


def run_speech():
    from signslator.demo_speech import main as speech_main
    speech_main()


def run_train(sign_name, samples=5, duration=3.0):
    from signslator.sign_to_speech import train_sign, list_trained_signs
    success = train_sign(sign_name, samples, duration)
    if success:
        list_trained_signs()


def run_list():
    from signslator.sign_to_speech import list_trained_signs
    list_trained_signs()


def run_recognize(threshold=0.3, max_dist=None, window=2.0):
    """max_dist=None means auto-calibrate from the templates."""
    from signslator.sign_to_speech import recognize_sign
    recognize_sign(threshold, max_dist, window)


def run_speech_to_sign():
    from signslator.speech_to_sign import main_loop
    main_loop()


def run_isl_download(words=None, user="ISL_DATA_USER001", sessions=4):
    """Download ISL500 signs and convert to templates."""
    from signslator.isl500_loader import download_and_convert, list_downloaded_signs
    download_and_convert(user, words, sessions)
    list_downloaded_signs()


def run_isl_evaluate():
    """Leave-one-out cross-validation on downloaded ISL500 templates."""
    import subprocess as sp
    script = os.path.join(PROJECT_ROOT, "signslator", "test_isl500_crossval.py")
    sp.run([sys.executable, script])


def run_sign_guide(sign=None, video=None, list_only=False):
    """Render sign animations (GIF) and optionally download reference videos."""
    import subprocess as sp
    script = os.path.join(PROJECT_ROOT, "signslator", "sign_guide.py")
    cmd = [sys.executable, script]
    if sign:
        cmd += ["--sign", sign]
    if video:
        cmd += ["--video", video]
    if list_only:
        cmd += ["--list"]
    sp.run(cmd)


def run_self_test(signs=None, samples=3):
    """Record your own samples and measure accuracy vs ISL500 and vs yourself."""
    import subprocess as sp
    script = os.path.join(PROJECT_ROOT, "signslator", "self_test.py")
    cmd = [sys.executable, script, "--samples", str(samples)]
    if signs:
        for s in signs:
            cmd += ["--sign", s]
    sp.run(cmd)


def run_chart(signs=None, cols=4, html_only=False):
    """Generate a visual reference chart of all signs."""
    import subprocess as sp
    script = os.path.join(PROJECT_ROOT, "signslator", "sign_chart.py")
    cmd = [sys.executable, script, "--cols", str(cols)]
    if html_only:
        cmd += ["--html-only"]
    if signs:
        for s in signs:
            cmd += ["--sign", s]
    sp.run(cmd)


def run_demo():
    """Quick demo: show camera landmarks briefly, then test speech."""
    import time
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import (
        HolisticLandmarker, HolisticLandmarkerOptions, RunningMode,
    )
    import numpy as np

    MODEL_PATH = os.path.join(PROJECT_ROOT, "signslator", "models", "holistic_landmarker.task")

    print("=" * 55)
    print("SignSlator Quick Demo")
    print("=" * 55)
    print()

    # 1. Camera + landmarks (3 seconds)
    print("[1/3] Camera + Landmarks — showing for 3 seconds...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
    )
    holistic = HolisticLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("  Camera not available — skipping visual demo")
    else:
        start = time.time()
        while time.time() - start < 3:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = holistic.detect_for_video(mp_image, int(time.time() * 1000))

            display = frame.copy()
            if result:
                pose_ok = result.pose_landmarks is not None
                l_hand = result.left_hand_landmarks is not None
                r_hand = result.right_hand_landmarks is not None
                status = f"Pose: {'OK' if pose_ok else 'no'} | L: {'Y' if l_hand else '-'} | R: {'Y' if r_hand else '-'}"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if result.left_hand_landmarks:
                    for lm in result.left_hand_landmarks:
                        x, y = int(lm.x * 640), int(lm.y * 480)
                        cv2.circle(display, (x, y), 4, (255, 0, 0), -1)
                if result.right_hand_landmarks:
                    for lm in result.right_hand_landmarks:
                        x, y = int(lm.x * 640), int(lm.y * 480)
                        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)

            cv2.imshow("SignSlator Demo — Camera", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("  ✓ Camera + landmarks working")
    holistic.close()

    # 2. Whisper test
    print("[2/3] Whisper speech recognition — loading model...")
    import whisper
    model = whisper.load_model("base")
    print("  ✓ Whisper loaded")

    # Synthetic audio test
    t = np.linspace(0, 1, 16000)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = model.transcribe(audio, language="en", fp16=False)
    print(f"  ✓ Whisper processed audio (result: '{result['text']}')")

    # 3. TTS test
    print("[3/3] Text-to-speech — testing...")
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    for v in voices:
        if 'Zira' in v.name:
            engine.setProperty('voice', v.id)
            break
    engine.say("SignSlator is ready.")
    engine.runAndWait()
    print("  ✓ TTS working")
    print()
    print("=" * 55)
    print("Demo complete! Next steps:")
    print("  1. Train signs:  python run.py train HELLO")
    print("  2. Recognize:    python run.py recognize")
    print("  3. Speech→Sign:  python run.py speech-to-sign")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="SignSlator — ISL Sign Language ↔ Speech")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Camera
    subparsers.add_parser("camera", help="Phase 1A: Camera + landmarks")

    # Speech
    subparsers.add_parser("speech", help="Phase 1B: Speech→text (Whisper)")

    # Train
    train_p = subparsers.add_parser("train", help="Phase 2: Train a sign")
    train_p.add_argument("sign", nargs="?", default=None, help="Sign name")
    train_p.add_argument("--samples", type=int, default=5)
    train_p.add_argument("--duration", type=float, default=3.0)

    # Recognize
    rec_p = subparsers.add_parser("recognize", help="Phase 2: Live recognition")
    rec_p.add_argument("--threshold", type=float, default=0.3)
    rec_p.add_argument("--max-dist", type=float, default=None,
                       help="Accept threshold (default: auto-calibrate from templates)")
    rec_p.add_argument("--window", type=float, default=2.0)

    # Speech to sign
    subparsers.add_parser("speech-to-sign", help="Phase 3: Speech→sign animation")

    # List
    subparsers.add_parser("list", help="List trained signs")

    # ISL500 download
    isl_p = subparsers.add_parser("isl-download", help="Download ISL500 signs (no recording needed)")
    isl_p.add_argument("--words", nargs="*", default=None,
                       help="Words to download (default: starter vocabulary)")
    isl_p.add_argument("--user", default="ISL_DATA_USER001",
                       help="ISL500 signer to use")
    isl_p.add_argument("--sessions", type=int, default=4,
                       help="Templates per sign (default: 4)")

    # ISL500 evaluate
    subparsers.add_parser("isl-evaluate", help="Cross-validate downloaded ISL500 signs")

    # Sign guide
    guide_p = subparsers.add_parser("guide", help="See what each sign looks like")
    guide_p.add_argument("--sign", help="Render one sign only")
    guide_p.add_argument("--video", metavar="SIGN",
                         help="Download the real reference video for a sign")
    guide_p.add_argument("--list", action="store_true",
                         help="List signs and which assets exist")

    # Self test
    self_p = subparsers.add_parser("self-test",
                                   help="Measure accuracy on your own hands")
    self_p.add_argument("--sign", action="append", default=None,
                        help="Sign(s) to test (repeatable)")
    self_p.add_argument("--samples", type=int, default=3,
                        help="Samples per sign (default 3)")

    # Chart
    chart_p = subparsers.add_parser("chart", help="Generate a visual sign chart")
    chart_p.add_argument("--sign", action="append", default=None,
                         help="Only these signs (repeatable)")
    chart_p.add_argument("--cols", type=int, default=4, help="Poster columns")
    chart_p.add_argument("--html-only", action="store_true",
                         help="Only write HTML, skip the PNG")

    # Demo
    subparsers.add_parser("demo", help="Quick demo of all components")

    args = parser.parse_args()

    if args.command == "camera":
        run_camera()
    elif args.command == "speech":
        run_speech()
    elif args.command == "train":
        if not args.sign:
            from signslator import STARTER_WORDS, get_vocabulary
            existing = get_vocabulary()
            if existing:
                print(f"Already have ({len(existing)}): {', '.join(existing)}")
            print("\nStarter words available from ISL500:")
            for i, s in enumerate(STARTER_WORDS, 1):
                print(f"  {i}. {s.upper()}")
            print("\nOr type any custom sign name.")
            sign = input("\nWhich sign to record? ").strip().upper()
            if sign:
                run_train(sign, args.samples, args.duration)
        else:
            run_train(args.sign, args.samples, args.duration)
    elif args.command == "recognize":
        run_recognize(args.threshold, args.max_dist, args.window)
    elif args.command == "speech-to-sign":
        run_speech_to_sign()
    elif args.command == "list":
        run_list()
    elif args.command == "isl-download":
        words = args.words
        if words is None:
            from signslator import STARTER_WORDS
            words = STARTER_WORDS
        run_isl_download(words, args.user, args.sessions)
    elif args.command == "isl-evaluate":
        run_isl_evaluate()
    elif args.command == "guide":
        run_sign_guide(args.sign, args.video, args.list)
    elif args.command == "self-test":
        run_self_test(args.sign, args.samples)
    elif args.command == "chart":
        run_chart(args.sign, args.cols, args.html_only)
    elif args.command == "demo":
        run_demo()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python run.py isl-download    # Get 13 ISL signs (no recording)")
        print("  python run.py isl-evaluate    # Check accuracy")
        print("  python run.py recognize       # Live recognition")
        print("  python run.py speech-to-sign  # Speech to animation")


if __name__ == "__main__":
    main()
