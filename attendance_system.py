# attendance_system.py — Pi optimized
import cv2
import numpy as np
import threading
import queue
from deepface import DeepFace
from ultralytics import YOLO
from datetime import datetime, date
from db_config import get_db, verify_connection

verify_connection()
db             = get_db()
students_col   = db["student_embeddings"]
attendance_col = db["attendance"]

RTSP_URL = "rtsp://admin:YOUR_PASSWORD@192.168.1.64:554/Streaming/Channels/102"

all_students = list(students_col.find({}, {"name":1, "faculty_no":1, "embedding":1}))
print(f"✅ Loaded {len(all_students)} students from Atlas")

# ── Load YOLO once at startup ─────────────────
model = YOLO("my_model.pt")

# ── Run YOLO on smaller input ─────────────────
PROCESS_WIDTH  = 320   # ← reduced from 640 (Pi runs ~2x faster)
PROCESS_HEIGHT = 240
THRESHOLD      = 0.6
FRAME_SKIP     = 3     # ← only run recognition every 3rd frame

# ── Thread-safe frame queue ───────────────────
# Separates camera reading from processing
# so camera never blocks and frames stay fresh
frame_queue    = queue.Queue(maxsize=1)
result_overlay = []    # stores last known boxes to draw between processed frames
marked_today   = set()

# ────────────────────────────────────────────────
#  Thread 1 — Camera reader
#  Runs continuously, always keeps latest frame
# ────────────────────────────────────────────────
def camera_reader():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Stream lost — reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # Drop old frame, keep only latest
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame)

# ────────────────────────────────────────────────
#  Cosine similarity
# ────────────────────────────────────────────────
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / norm if norm > 0 else 0.0

# ────────────────────────────────────────────────
#  ArcFace recognition
# ────────────────────────────────────────────────
def recognize_face(face_crop):
    try:
        result = DeepFace.represent(
            img_path          = face_crop,
            model_name        = "ArcFace",
            enforce_detection = False,
            detector_backend  = "opencv"   # fastest backend on Pi
        )
        query_emb  = result[0]["embedding"]
        best_match = None
        best_score = -1

        for student in all_students:
            score = cosine_similarity(query_emb, student["embedding"])
            if score > best_score:
                best_score = score
                best_match = student

        if best_score >= THRESHOLD:
            return best_match["name"], best_match["faculty_no"], round(best_score, 3)
        return "Unknown", None, round(best_score, 3)
    except:
        return "Unknown", None, 0.0

# ────────────────────────────────────────────────
#  Mark attendance in Atlas
# ────────────────────────────────────────────────
def mark_attendance(name, faculty_no):
    today = str(date.today())
    if not attendance_col.find_one({"faculty_no": faculty_no, "date": today}):
        attendance_col.insert_one({
            "name"       : name,
            "faculty_no" : faculty_no,
            "date"       : today,
            "timestamp"  : datetime.utcnow(),
            "status"     : "Present"
        })
        print(f"✅ Marked: {name} ({faculty_no})")
        return True
    return False

# ────────────────────────────────────────────────
#  Start camera thread
# ────────────────────────────────────────────────
t = threading.Thread(target=camera_reader, daemon=True)
t.start()
print("📷 Camera thread started...")

# ────────────────────────────────────────────────
#  Main loop — detection + recognition
# ────────────────────────────────────────────────
frame_count   = 0
overlay_boxes = []   # drawn every frame even when skipping recognition

print("Running... Press Q to quit\n")

while True:
    # Get latest frame from camera thread
    try:
        frame = frame_queue.get(timeout=5)
    except queue.Empty:
        print("⚠ No frame received in 5s")
        continue

    frame_count += 1

    # Always resize for display
    display = cv2.resize(frame, (640, 480))
    dh, dw  = display.shape[:2]

    # ── Only run YOLO + ArcFace every Nth frame ──
    if frame_count % FRAME_SKIP == 0:

        # Shrink for YOLO (much faster on Pi)
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        scale_x = dw / PROCESS_WIDTH
        scale_y = dh / PROCESS_HEIGHT

        results    = model(small, verbose=False)
        detections = results[0].boxes
        overlay_boxes = []   # reset

        for det in detections:
            conf = det.conf.item()
            if conf < 0.45:
                continue

            xyxy_np = det.xyxy.cpu().numpy()
            xyxy    = xyxy_np.squeeze().astype(int)
            if xyxy.shape != (4,):
                continue

            # Scale coords back to display size
            xmin = int(xyxy[0] * scale_x)
            ymin = int(xyxy[1] * scale_y)
            xmax = int(xyxy[2] * scale_x)
            ymax = int(xyxy[3] * scale_y)

            # Crop face from display-size frame for recognition
            x1 = max(0, xmin - 10)
            y1 = max(0, ymin - 10)
            x2 = min(dw, xmax + 10)
            y2 = min(dh, ymax + 10)
            face_crop = display[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            name, faculty_no, score = recognize_face(face_crop)

            if name != "Unknown" and faculty_no not in marked_today:
                if mark_attendance(name, faculty_no):
                    marked_today.add(faculty_no)

            # Store box for drawing on skipped frames too
            overlay_boxes.append({
                "box"        : (xmin, ymin, xmax, ymax),
                "name"       : name,
                "faculty_no" : faculty_no,
                "score"      : score,
                "marked"     : faculty_no in marked_today if faculty_no else False
            })

    # ── Draw stored boxes on every frame ─────────
    for item in overlay_boxes:
        xmin, ymin, xmax, ymax = item["box"]
        name       = item["name"]
        score      = item["score"]
        is_marked  = item["marked"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({score})"
        if is_marked:
            label += " [Present]"

        cv2.rectangle(display, (xmin, ymin), (xmax, ymax), color, 2)

        # Label background box
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (xmin, ymin-lh-8), (xmin+lw+4, ymin), color, cv2.FILLED)
        cv2.putText(display, label, (xmin+2, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # ── HUD ───────────────────────────────────────
    cv2.putText(display, f"Present: {len(marked_today)}/{len(all_students)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(display, str(date.today()),
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(display, f"Frame: {frame_count} | Skip: every {FRAME_SKIP}",
                (10, dh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

    cv2.imshow("Attendance System", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\nDone. Total marked today: {len(marked_today)}")