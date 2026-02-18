"""
capture_faces.py - Register new employees by capturing face images for training

Purpose:
    Captures approximately 50 grayscale face images per user via webcam using
    Haar Cascade for face detection. Images are saved under dataset/<employee_id>/
    for use by train_model.py. Run this script when registering a new employee.

Usage:
    Run from project root: python -m biometric_module.capture_faces
    Or: python biometric_module/capture_faces.py (from project root)
"""

import os
import sys
import cv2

# Add project root to path so we can import from biometric_module
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from biometric_module.utils import (
    ensure_directories,
    get_face_cascade,
    DATASET_DIR,
    IMAGES_PER_USER,
)


def capture_face_images(employee_id: int, num_images: int = IMAGES_PER_USER) -> bool:
    """
    Capture face images for a single employee and save to dataset/<employee_id>/.

    Args:
        employee_id: Unique integer ID for the employee (used as folder name and label).
        num_images: Number of face images to capture (default from utils.IMAGES_PER_USER).

    Returns:
        True if capture completed successfully, False if cancelled or error.
    """
    ensure_directories()
    user_dir = os.path.join(DATASET_DIR, str(employee_id))
    os.makedirs(user_dir, exist_ok=True)

    try:
        face_cascade = get_face_cascade()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    # Try to open default camera (0); on some systems use 1 or 2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera connection and permissions.")
        return False

    print(f"[INFO] Camera opened. Capturing {num_images} images for Employee ID: {employee_id}")
    print("[INFO] Look at the camera. Keep your face in the green rectangle.")
    print("[INFO] Press SPACE to capture a frame, or Q to quit early.")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle and status on frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Use the first (largest) face only for capture
            if count < num_images:
                face_roi = gray[y : y + h, x : x + w]
                # Save when SPACE is pressed and a face is detected
                pass  # we capture on key press below

        # Show count and instructions
        cv2.putText(
            frame,
            f"Captured: {count}/{num_images} - SPACE to capture, Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Face detected - press SPACE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("[INFO] Capture cancelled by user.")
            break
        if key == ord(" "):
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = gray[y : y + h, x : x + w]
                img_path = os.path.join(user_dir, f"{employee_id}_{count:04d}.jpg")
                cv2.imwrite(img_path, face_roi)
                count += 1
                print(f"  Saved image {count}/{num_images}")
            elif len(faces) > 1:
                print("[WARN] Multiple faces detected. Please ensure only one person is in frame.")
            else:
                print("[WARN] No face detected. Move into frame and try again.")

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"[SUCCESS] Saved {count} images to {user_dir}")
        return True
    print("[INFO] No images saved.")
    return False


def run_auto_capture(employee_id: int, num_images: int = IMAGES_PER_USER) -> bool:
    """
    Alternative: auto-capture images at intervals when a single face is detected.
    Useful for faster registration without pressing SPACE each time.
    """
    ensure_directories()
    user_dir = os.path.join(DATASET_DIR, str(employee_id))
    os.makedirs(user_dir, exist_ok=True)

    try:
        face_cascade = get_face_cascade()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return False

    print(f"[INFO] Auto-capturing {num_images} images for Employee ID: {employee_id}")
    print("[INFO] Keep one face in frame. Press Q to quit.")

    count = 0
    frame_skip = 5  # Capture every N frames to get variety
    frame_counter = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(faces) == 1 and frame_counter % frame_skip == 0:
            x, y, w, h = faces[0]
            face_roi = gray[y : y + h, x : x + w]
            img_path = os.path.join(user_dir, f"{employee_id}_{count:04d}.jpg")
            cv2.imwrite(img_path, face_roi)
            count += 1
            if count % 10 == 0:
                print(f"  Captured {count}/{num_images}")

        cv2.putText(
            frame,
            f"Auto-capture: {count}/{num_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Saved {count} images to {user_dir}")
    return count > 0


def main():
    """CLI entry: prompt for employee ID and run capture."""
    print("=== Face Capture - Register New Employee ===\n")
    try:
        emp_id_str = input("Enter Employee ID (integer): ").strip()
        employee_id = int(emp_id_str)
    except ValueError:
        print("[ERROR] Employee ID must be an integer.")
        return
    mode = input("Auto-capture? (y/n, default n): ").strip().lower()
    if mode == "y":
        run_auto_capture(employee_id)
    else:
        capture_face_images(employee_id)


if __name__ == "__main__":
    main()
