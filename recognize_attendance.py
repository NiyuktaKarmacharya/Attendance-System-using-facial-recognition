"""
recognize_attendance.py - Real-time attendance via face recognition + liveness + DB

Purpose:
    Uses webcam feed to detect faces (Haar), recognize identity (LBPH), run
    liveness check (head movement), then records clock_in or clock_out in the
    database. First recognition of the day = clock_in; second = clock_out.
    Prevents duplicate entries. Attendance is only marked after liveness is confirmed.

Usage:
    Run from project root: python -m biometric_module.recognize_attendance
"""

import os
import sys
import cv2
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from biometric_module.utils import (
    ensure_directories,
    get_face_cascade,
    TRAINER_FILE,
)
from biometric_module.liveness_detection import LivenessChecker, run_liveness_check
from database.db import get_connection, get_today_status, record_clock_in, record_clock_out


# Confidence threshold: LBPH returns distance; lower is better. Typical good match < 80.
CONFIDENCE_THRESHOLD = 80
# Cooldown (seconds) after marking attendance before allowing same person again (avoid double-trigger)
ATTENDANCE_COOLDOWN_SEC = 3.0
# Frames to collect for liveness before checking
LIVENESS_FRAME_COUNT = 25


def load_recognizer():
    """Load LBPH model from trainer/trainer.yml. Returns (recognizer, True) or (None, False)."""
    if not os.path.isfile(TRAINER_FILE):
        print(f"[ERROR] Model not found: {TRAINER_FILE}. Run train_model.py first.")
        return None, False
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(TRAINER_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, False
    return recognizer, True


def run_attendance_system():
    """
    Main loop: webcam -> detect face -> recognize -> liveness -> record clock_in/clock_out.
    """
    ensure_directories()
    face_cascade = get_face_cascade()
    recognizer, ok = load_recognizer()
    if not ok:
        return

    conn = get_connection()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Attendance system started. Look at camera to clock in/out. Press Q to quit.")
    print("[INFO] Liveness check: move your head slightly when prompted.\n")

    liveness_checker = LivenessChecker()
    last_attendance_time = 0
    current_employee_id = None  # track who we're collecting liveness for
    liveness_centroids = []
    state = "detect"  # detect -> collect_liveness -> confirm

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        # Single face: try to recognize
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_roi = gray[y : y + h, x : x + w]
            emp_id, confidence = recognizer.predict(face_roi)
            centroid = (x + w // 2, y + h // 2)

            if state == "detect":
                if confidence < CONFIDENCE_THRESHOLD:
                    # Recognized: start collecting liveness for this person
                    current_employee_id = emp_id
                    liveness_centroids = [centroid]
                    state = "collect_liveness"
                    print(f"[INFO] Recognized Employee {emp_id}. Please move your head slightly...")
                else:
                    cv2.putText(
                        frame,
                        "Unknown - not registered",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            elif state == "collect_liveness" and current_employee_id == emp_id:
                liveness_centroids.append(centroid)
                if len(liveness_centroids) >= LIVENESS_FRAME_COUNT:
                    # Check if head moved enough
                    from biometric_module.liveness_detection import check_liveness_head_movement
                    if check_liveness_head_movement(liveness_centroids):
                        # Liveness OK: record attendance
                        status = get_today_status(conn, current_employee_id)
                        now = time.time()
                        if now - last_attendance_time < ATTENDANCE_COOLDOWN_SEC:
                            pass  # still in cooldown, don't record again
                        elif status is None or status.get("clock_in") is None:
                            if record_clock_in(conn, current_employee_id):
                                print(f"[ATTENDANCE] Employee {current_employee_id} clocked IN.")
                                last_attendance_time = now
                        elif status.get("clock_out") is None:
                            if record_clock_out(conn, current_employee_id):
                                print(f"[ATTENDANCE] Employee {current_employee_id} clocked OUT.")
                                last_attendance_time = now
                        else:
                            cv2.putText(
                                frame,
                                "Already clocked in & out today",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2,
                            )
                        state = "detect"
                        current_employee_id = None
                        liveness_centroids = []
                    else:
                        # Not enough movement: reset and ask again
                        liveness_centroids = []
                        print("[LIVENESS] Movement not detected. Please try again.")
                else:
                    cv2.putText(
                        frame,
                        f"Move head... {len(liveness_centroids)}/{LIVENESS_FRAME_COUNT}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            # Draw face box and label
            color = (0, 255, 0) if confidence < CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"Emp {emp_id} ({confidence:.0f})" if confidence < 100 else "Unknown"
            if state == "detect":
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            if state != "detect":
                state = "detect"
                current_employee_id = None
                liveness_centroids = []

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("[INFO] Attendance system stopped.")


def main():
    """CLI entry."""
    print("=== Biometric Attendance - Recognition ===\n")
    run_attendance_system()


if __name__ == "__main__":
    main()
