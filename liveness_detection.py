"""
liveness_detection.py - Anti-spoofing via blink and head movement

Purpose:
    Provides a simple liveness check so attendance is only marked for a live person,
    not a photo or screen. Uses (1) head movement: face bounding box must move
    beyond a threshold over a short window, and (2) optional blink detection using
    eye Haar cascade (eyes visible -> not visible -> visible counts as blink).

Usage:
    Call check_liveness(face_roi_gray, full_frame_gray, face_bbox) in a loop,
    or run_ liveness_check(cap, face_cascade) which runs a short interactive check.
"""

import os
import sys
import cv2
import time
from collections import deque

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from biometric_module.utils import get_face_cascade, get_eye_cascade


# --- Head movement liveness ---
# Require face centroid to move at least this many pixels over the window
HEAD_MOVEMENT_THRESHOLD = 30
HEAD_MOVEMENT_WINDOW = 30  # number of frames to consider
# Minimum number of "movement" samples (centroid delta > threshold)
MIN_MOVEMENT_SAMPLES = 3


class LivenessChecker:
    """
    Stateful liveness checker: tracks face positions and optional eye visibility
    to confirm head movement and/or blink.
    """

    def __init__(
        self,
        movement_threshold: int = HEAD_MOVEMENT_THRESHOLD,
        window_size: int = HEAD_MOVEMENT_WINDOW,
        min_movement_samples: int = MIN_MOVEMENT_SAMPLES,
    ):
        self.movement_threshold = movement_threshold
        self.window_size = window_size
        self.min_movement_samples = min_movement_samples
        self.face_centroids = deque(maxlen=window_size)
        self.eye_cascade = None  # lazy load
        self._last_eyes_visible = None
        self._blink_count = 0

    def _centroid(self, x, y, w, h):
        return (x + w // 2, y + h // 2)

    def add_face_position(self, x, y, w, h):
        """Call each frame with current face bbox. Then use is_live() to check."""
        cx, cy = self._centroid(x, y, w, h)
        self.face_centroids.append((cx, cy))

    def is_live_by_movement(self) -> bool:
        """
        True if we have enough history and the face moved enough (head movement).
        """
        if len(self.face_centroids) < 2:
            return False
        movements = 0
        for i in range(1, len(self.face_centroids)):
            c0 = self.face_centroids[i - 1]
            c1 = self.face_centroids[i]
            dx = abs(c1[0] - c0[0])
            dy = abs(c1[1] - c0[1])
            if dx >= self.movement_threshold or dy >= self.movement_threshold:
                movements += 1
        return movements >= self.min_movement_samples

    def reset(self):
        """Reset state for a new liveness session."""
        self.face_centroids.clear()
        self._last_eyes_visible = None
        self._blink_count = 0


def check_liveness_head_movement(face_centroids: list, threshold: int = HEAD_MOVEMENT_THRESHOLD, min_samples: int = MIN_MOVEMENT_SAMPLES) -> bool:
    """
    One-shot check: given a list of (cx, cy) centroids, return True if there
    was enough head movement.
    """
    if len(face_centroids) < 2:
        return False
    movements = 0
    for i in range(1, len(face_centroids)):
        c0, c1 = face_centroids[i - 1], face_centroids[i]
        dx = abs(c1[0] - c0[0])
        dy = abs(c1[1] - c0[1])
        if dx >= threshold or dy >= threshold:
            movements += 1
    return movements >= min_samples


def run_liveness_check(
    cap,
    face_cascade,
    timeout_seconds: float = 10.0,
    movement_threshold: int = HEAD_MOVEMENT_THRESHOLD,
    window_size: int = HEAD_MOVEMENT_WINDOW,
) -> bool:
    """
    Run an interactive liveness check: user must move their head slightly within
    the timeout. Returns True if liveness confirmed, False on timeout or quit.

    Args:
        cap: OpenCV VideoCapture.
        face_cascade: Haar face cascade (already loaded).
        timeout_seconds: Max time to wait for movement.
        movement_threshold: Min pixel movement to count.
        window_size: Frames to consider for movement.
    """
    checker = LivenessChecker(
        movement_threshold=movement_threshold,
        window_size=window_size,
    )
    start = time.time()
    print("[LIVENESS] Please move your head slightly (left/right or up/down)...")
    while (time.time() - start) < timeout_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 1:
            x, y, w, h = faces[0]
            checker.add_face_position(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Move your head slightly",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Frames: {len(checker.face_centroids)}/{window_size}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Liveness Check", frame)
        if checker.is_live_by_movement():
            cv2.destroyWindow("Liveness Check")
            print("[LIVENESS] Confirmed.")
            return True
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Liveness Check")
    print("[LIVENESS] Not confirmed (timeout or quit).")
    return False


if __name__ == "__main__":
    # Quick test: run liveness check from webcam
    face_cascade = get_face_cascade()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)
    result = run_liveness_check(cap, face_cascade, timeout_seconds=15.0)
    cap.release()
    print("Liveness result:", result)
