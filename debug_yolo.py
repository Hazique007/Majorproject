import cv2
import numpy as np
import threading
import time
from deepface import DeepFace
from ultralytics import YOLO
from db_config import get_db, verify_connection

verify_connection()
db = get_db()
all_students = list(db["student_embeddings"].find({}, {"name":1,"faculty_no":1,"embedding":1}))
print(f"✅ Loaded {len(all_students)} students\n")

model     = YOLO("my_model.pt")
THRESHOLD = 0.60
PADDING   = 30

CAMERA_IP = "192.168.1.64"
USERNAME  = "admin"
PASSWORD  = "JABIN002@yunus"
RTSP_URL  = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/102"

latest_frame  = None
frame_lock    = threading.Lock()
overlay_boxes = []
boxes_lock    = threading.Lock()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / norm if norm > 0 else 0.0

def recognize(face_crop):
    try:
        result = DeepFace.represent(
            img_path          = face_crop,
            model_name        = "ArcFace",
            enforce_detection = False,
            detector_backend  = "opencv"
        )
        query_emb = result[0]["embedding"]
        scores = []
        for student in all_students:
            score = cosine_similarity(query_emb, student["embedding"])
            scores.append((score, student["name"], student["faculty_no"]))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_name, best_fno = scores[0]

        if best_score >= THRESHOLD:
            return best_name, best_fno, round(best_score, 3)
        return "Unknown", None, round(best_score, 3)
    except Exception as e:
        print(f"  ⚠ {e}")
        return "Unknown", None, 0.0

def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        with frame_lock:
            latest_frame = frame.copy()

def recognition_thread():
    global overlay_boxes
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        h, w   = frame.shape[:2]
        small  = cv2.resize(frame, (416, 416))
        sx, sy = w/416, h/416

        results    = model(small, verbose=False)
        detections = results[0].boxes
        boxes      = []

        for det in detections:
            conf    = det.conf.item()
            xyxy_np = det.xyxy.cpu().numpy()
            if xyxy_np.ndim > 1:
                xyxy_np = xyxy_np.squeeze()
            xyxy_np = xyxy_np.astype(int)
            if xyxy_np.shape != (4,) or conf < 0.35:
                continue

            xmin = int(xyxy_np[0] * sx)
            ymin = int(xyxy_np[1] * sy)
            xmax = int(xyxy_np[2] * sx)
            ymax = int(xyxy_np[3] * sy)

            x1 = max(0, xmin - PADDING)
            y1 = max(0, ymin - PADDING)
            x2 = min(w, xmax + PADDING)
            y2 = min(h, ymax + PADDING)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            fh, fw = face_crop.shape[:2]
            if fw < 112 or fh < 112:
                face_crop = cv2.resize(face_crop, (112,112),
                                       interpolation=cv2.INTER_CUBIC)

            name, fno, score = recognize(face_crop)
            print(f"  → {name} | {score}")
            boxes.append({
                "box"  : (xmin,ymin,xmax,ymax),
                "name" : name,
                "score": score
            })

        with boxes_lock:
            overlay_boxes = boxes

        time.sleep(0.1)

threading.Thread(target=camera_thread,      daemon=True).start()
threading.Thread(target=recognition_thread, daemon=True).start()

print("⏳ Warming up...")
time.sleep(2)
print("✅ Running! Press Q to quit\n")

while True:
    with frame_lock:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        display = latest_frame.copy()

    with boxes_lock:
        boxes = list(overlay_boxes)

    for item in boxes:
        xmin,ymin,xmax,ymax = item["box"]
        name  = item["name"]
        score = item["score"]
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        label = f"{name} ({score})"
        cv2.rectangle(display, (xmin,ymin), (xmax,ymax), color, 2)
        (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (xmin,ymin-lh-12), (xmin+lw+4,ymin), color, cv2.FILLED)
        cv2.putText(display, label, (xmin+2,ymin-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    cv2.putText(display, f"Students: {len(all_students)} | Faces: {len(boxes)}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
    cv2.putText(display, f"Threshold: {THRESHOLD}",
                (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(display, "Q = quit",
                (10,82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("HIKVision Recognition", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()