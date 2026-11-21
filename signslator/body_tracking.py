"""
Head + torso tracking, drawn in pink.

Uses MediaPipe's PoseLandmarker alongside the HandLandmarker rather than going
back to Holistic: the hand model is kept because its landmarks are 2.2x steadier
(measured), and a second small model costs only ~12 ms/frame.

Only the head (landmarks 0-10) and torso (shoulders 11/12, hips 23/24) are drawn
— arms, legs and feet are detected but not rendered.

Note: this is a live overlay. Recorded templates store hand features only, so
replayed animations do not include the body.
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models",
                               "pose_landmarker_lite.task")

# Pink palette (BGR).
PINK_LINE = (190, 110, 255)
PINK_DOT = (235, 170, 255)
PINK_TEXT = (200, 140, 255)

# Pose landmark indices (MediaPipe's 33-point scheme).
HEAD_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # nose, eyes, ears, mouth
TORSO_IDX = [11, 12, 23, 24]                    # shoulders and hips

HEAD_CONN = [
    (1, 2), (2, 3),              # left eye
    (4, 5), (5, 6),              # right eye
    (2, 5),                      # brow
    (1, 4),                      # inner eye corners
    (3, 7),                      # left temple -> left ear
    (6, 8),                      # right temple -> right ear
    (7, 9), (9, 10), (10, 8),    # mouth and jaw corners
]

TORSO_CONN = [
    (11, 12),                    # shoulders
    (11, 23),                    # left flank
    (12, 24),                    # right flank
    (23, 24),                    # hips
]

_pose_landmarker = None
_timestamp = [0]


def get_pose_landmarker():
    """Create the pose landmarker on first use."""
    global _pose_landmarker
    if _pose_landmarker is None:
        if not os.path.exists(POSE_MODEL_PATH):
            raise FileNotFoundError(
                f"Pose model missing: {POSE_MODEL_PATH}\n"
                "Download it with:  python setup.py"
            )
        base = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = PoseLandmarkerOptions(
            base_options=base,
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        _pose_landmarker = PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def next_timestamp() -> int:
    """Monotonically increasing millisecond timestamp for VIDEO mode."""
    ts = int(time.time() * 1000)
    if ts <= _timestamp[0]:
        ts = _timestamp[0] + 1
    _timestamp[0] = ts
    return ts


def detect_pose(mp_image):
    """Run pose detection on an mp.Image. Returns the landmarker result."""
    return get_pose_landmarker().detect_for_video(mp_image, next_timestamp())


def draw_body(display, pose_landmarks, label: bool = True) -> int:
    """
    Draw head and torso on `display` in pink. Returns landmarks drawn (0 if none).

    VIDEO mode returns one list per detected pose; IMAGE mode returns a flat list.
    Both are accepted.

    Call this BEFORE drawing the hands so the hands stay on top.
    """
    if not pose_landmarks:
        return 0

    # VIDEO mode: outer list per pose. Take the first (strongest) pose.
    if isinstance(pose_landmarks[0], (list, tuple)):
        pose_landmarks = pose_landmarks[0]
    if not pose_landmarks:
        return 0

    n_pts = len(pose_landmarks)
    if n_pts <= max(TORSO_IDX):
        return 0

    h, w = display.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]

    for i, j in TORSO_CONN:
        cv2.line(display, pts[i], pts[j], PINK_LINE, 3)
    for i, j in HEAD_CONN:
        cv2.line(display, pts[i], pts[j], PINK_LINE, 2)

    for i in TORSO_IDX:
        cv2.circle(display, pts[i], 7, PINK_DOT, -1)
    for i in HEAD_IDX:
        cv2.circle(display, pts[i], 5, PINK_DOT, -1)

    if label:
        cv2.putText(display, "HEAD + TORSO", (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PINK_TEXT, 1)

    return n_pts
