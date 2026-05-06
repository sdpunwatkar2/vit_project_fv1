import cv2
import os
import time
import glob
import sqlite3
import shutil
import subprocess
import platform
from datetime import datetime
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

LOG_DIR = "logs"
SNAPSHOT_DIR = "outputs/snapshots"
MODEL_PATH = "yolov8n.pt"

CAMERA_HOST = os.getenv("CAMERA_HOST", "0")  # "0" = webcam
print(f"[DEBUG] CAMERA_HOST = {CAMERA_HOST}")

CAMERA_FLIP_MODE = 1
SNAPSHOT_RETENTION_DAYS = 7
LATEST_FRAME_PATH = "latest.jpg"
ALERT_COOLDOWN_SEC = 5

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# =========================
# UTIL FUNCTIONS
# =========================

def load_model():
    model = YOLO(MODEL_PATH)
    class_id = None
    for idx, name in model.names.items():
        if name == "dog":
            class_id = idx
            break
    if class_id is None:
        raise ValueError("Dog class not found in model")
    return model, class_id


def init_db():
    conn = sqlite3.connect(os.path.join(LOG_DIR, "detections.db"))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            speed REAL,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    return conn, c


def normalize_frame(frame):
    if CAMERA_FLIP_MODE is None:
        return frame
    return cv2.flip(frame, CAMERA_FLIP_MODE)


def cleanup_old_snapshots():
    cutoff = time.time() - (SNAPSHOT_RETENTION_DAYS * 86400)
    for f in glob.glob(os.path.join(SNAPSHOT_DIR, "*")):
        if os.path.getmtime(f) < cutoff:
            os.remove(f)


def find_ip_camera():
    endpoints = [
        f"http://{CAMERA_HOST}/video",
        f"http://{CAMERA_HOST}/stream",
        f"http://{CAMERA_HOST}/shot.jpg",
    ]

    for ep in endpoints:
        print(f"[INFO] Trying {ep}")
        cap = cv2.VideoCapture(ep)
        time.sleep(1)
        ret, frame = cap.read()
        if ret:
            print(f"[INFO] Connected to {ep}")
            return cap, ep
        cap.release()

    return None, None


def play_sound():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 300)
    else:
        print("\a")


# =========================
# MAIN
# =========================

def main():

    # CAMERA INIT
    if CAMERA_HOST == "0":
        print("[INFO] Using webcam")
        cap = cv2.VideoCapture(0)
        used_endpoint = "webcam"
    else:
        cap, used_endpoint = find_ip_camera()
        if cap is None:
            print("[ERROR] Could not connect to camera")
            return

    time.sleep(1)

    # MODEL + DB
    model, DOG_CLASS = load_model()
    conn, c = init_db()

    tracker = {}
    last_alert = {}

    print("[INFO] Running... Press Q to exit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Frame failed, reconnecting...")

            if CAMERA_HOST == "0":
                cap = cv2.VideoCapture(0)
            else:
                cap, _ = find_ip_camera()

            continue

        frame = normalize_frame(frame)

        results = model.track(
            frame,
            persist=True,
            classes=[DOG_CLASS],
            conf=0.35,
            verbose=False
        )

        annotated = frame.copy()

        if results and results[0].boxes is not None:
            annotated = results[0].plot()

            for r in results:
                if r.boxes.id is None:
                    continue

                for box, cls_id, track_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):

                    if int(cls_id) != DOG_CLASS:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    now = datetime.now()

                    track_id = int(track_id)

                    speed = 0
                    if track_id in tracker:
                        px, py, pt = tracker[track_id]
                        dist = ((cx-px)**2 + (cy-py)**2)**0.5
                        dt = (now - pt).total_seconds()
                        speed = dist / (dt+1e-6)

                    tracker[track_id] = (cx, cy, now)

                    # SNAPSHOT
                    snap_path = os.path.join(SNAPSHOT_DIR, f"dog_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, frame[y1:y2, x1:x2])

                    # DB
                    c.execute(
                        "INSERT INTO logs VALUES (?, ?, ?)",
                        (now.strftime("%Y-%m-%d %H:%M:%S"), speed, snap_path)
                    )
                    conn.commit()

                    print(f"[INFO] Dog detected | Speed={round(speed,2)}")

                    # SOUND
                    if track_id not in last_alert or time.time() - last_alert[track_id] > ALERT_COOLDOWN_SEC:
                        play_sound()
                        last_alert[track_id] = time.time()

        cv2.imwrite(LATEST_FRAME_PATH, annotated)
        cv2.imshow("Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cleanup_old_snapshots()

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    main()