"""
utils.py - Shared utilities for the Biometric Attendance System

Purpose:
    Provides path resolution, cascade classifier loading, and common constants
    used across capture_faces, train_model, recognize_attendance, and liveness_detection.
    Uses relative paths so the project works on Windows, macOS, and Linux.
"""
import os
import sys


# ---------------------------------------------------------------------------
# Path setup: ensure we can import and resolve paths from project root
# ---------------------------------------------------------------------------
def get_project_root():
    """Return the absolute path to the attendance_system project root (works from any cwd)."""
    # Derive from this file's location: utils.py is in attendance_system/biometric_module/
    this_file = os.path.abspath(__file__)
    biometric_dir = os.path.dirname(this_file)
    root = os.path.dirname(biometric_dir)
    return root


PROJECT_ROOT = get_project_root()

# Key directories (created on first use if missing)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAINER_DIR = os.path.join(PROJECT_ROOT, "trainer")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "attendance.db")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")

# Number of face images to capture per user during registration
IMAGES_PER_USER = 50

# ---------------------------------------------------------------------------
# Cascade classifiers (OpenCV Haar) - cv2 imported on first use
# ---------------------------------------------------------------------------
def get_face_cascade():
    """Load and return the Haar cascade for frontal face detection."""
    import cv2
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise FileNotFoundError("Could not load haarcascade_frontalface_default.xml")
    return cascade


def get_eye_cascade():
    """Load and return the Haar cascade for eye detection (used in liveness)."""
    import cv2
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    if cascade.empty():
        raise FileNotFoundError("Could not load haarcascade_eye.xml")
    return cascade


def ensure_directories():
    """Create dataset, trainer, and database directories if they do not exist."""
    for dir_path in (DATASET_DIR, TRAINER_DIR, DATABASE_DIR):
        os.makedirs(dir_path, exist_ok=True)
