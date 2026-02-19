"""
train_model.py - Train LBPH face recognizer on captured dataset

Purpose:
    Reads grayscale face images from dataset/<employee_id>/ and trains OpenCV's
    LBPH (Local Binary Pattern Histograms) face recognizer. Saves the trained
    model to trainer/trainer.yml for use by recognize_attendance.py.

Usage:
    Run from project root: python -m biometric_module.train_model
"""

import os
import sys
import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from biometric_module.utils import (
    ensure_directories,
    DATASET_DIR,
    TRAINER_DIR,
    TRAINER_FILE,
)


def get_images_and_labels():
    """
    Walk dataset/ folder: each subdir name is employee_id (label), files are face images.
    Returns (faces, labels) as lists of numpy arrays and ints.
    """
    faces = []
    labels = []
    if not os.path.isdir(DATASET_DIR):
        return faces, labels

    for dir_name in sorted(os.listdir(DATASET_DIR)):
        dir_path = os.path.join(DATASET_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue
        try:
            employee_id = int(dir_name)
        except ValueError:
            continue
        for fname in os.listdir(dir_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(dir_path, fname)
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(employee_id)
            except Exception:
                continue
    return faces, labels


def train_lbph(save_path: str = None) -> bool:
    """
    Train LBPH recognizer on dataset and save to trainer/trainer.yml.

    Args:
        save_path: Path to save trainer.yml; defaults to utils.TRAINER_FILE.

    Returns:
        True if training succeeded and model was saved, False otherwise.
    """
    ensure_directories()
    save_path = save_path or TRAINER_FILE
    faces, labels = get_images_and_labels()

    if len(faces) == 0:
        print("[ERROR] No face images found in dataset/. Run capture_faces.py first.")
        return False

    print(f"[INFO] Found {len(faces)} images for {len(set(labels))} employee(s). Training LBPH...")
    labels = np.array(labels, dtype=np.int32)
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )
    recognizer.train(faces, labels)
    recognizer.write(save_path)
    print(f"[SUCCESS] Model saved to {save_path}")
    return True


def main():
    """CLI entry: run training and save model."""
    print("=== Train LBPH Face Model ===\n")
    train_lbph()


if __name__ == "__main__":
    main()
