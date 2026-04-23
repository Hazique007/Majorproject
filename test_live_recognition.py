# test_live_recognition.py
import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from db_config import get_db, verify_connection

verify_connection()
db = get_db()
all_students = list(db["student_embeddings"].find({}, {"name":1, "faculty_no":1, "embedding":1}))
print(f"✅ Loaded {len(all_students)} students\n")

model     = YOLO("my_model.pt")
THRESHOLD = 0.55  # slightly relaxed for webcam lighting
PADDING   = 30

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
        query_emb  = result[0]["embedding"]
        best_name  = "Unknown"
        best_score = -1
        for student in all_students:
            score = cosine_similarity(query_emb, student["embedding"])
            if score > best_score:
                best_score = score
                best_name  = student["name"] if score >= THRESHOLD else "Unknown"
        return best_name, round(best_score, 3)
    except:
        return "Unknown", 0.0

cap = cv2.VideoCapture(0)
print("📷 Webcam started — press Q to quit\n")

frame_count   = 0

# ── These persist across frames so box never disappears ──
overlay_boxes = []   # ← KEY FIX: draw last known boxes every frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    h, w    = display.shape[:2]
    frame_count += 1

    # Run YOLO + ArcFace every 5th frame
    if frame_count % 5 == 0:
        results    = model(frame, verbose=False)
        detections = results[0].boxes
        overlay_boxes = []  # reset

        for det in detections:
            conf    = det.conf.item()
            xyxy_np = det.xyxy.cpu().numpy()

            # Safe squeeze
            if xyxy_np.ndim > 1:
                xyxy_np = xyxy_np.squeeze()
            xyxy_np = xyxy_np.astype(int)

            if xyxy_np.shape != (4,) or conf < 0.35:  # lowered YOLO conf too
                continue

            xmin, ymin, xmax, ymax = xyxy_np
            x1 = max(0, xmin - PADDING)
            y1 = max(0, ymin - PADDING)
            x2 = min(w, xmax + PADDING)
            y2 = min(h, ymax + PADDING)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            name, score = recognize(face_crop)
            print(f"  → {name} | score: {score}")

            overlay_boxes.append({
                "box"  : (xmin, ymin, xmax, ymax),
                "name" : name,
                "score": score
            })

    # ── Draw on EVERY frame (not just every 5th) ──
    for item in overlay_boxes:
        xmin, ymin, xmax, ymax = item["box"]
        name  = item["name"]
        score = item["score"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({score})"

        cv2.rectangle(display, (xmin, ymin), (xmax, ymax), color, 2)

        # Label with background so it's readable
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (xmin, ymin-lh-12), (xmin+lw+4, ymin), color, cv2.FILLED)
        cv2.putText(display, label, (xmin+2, ymin-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # HUD
    status = f"Faces detected: {len(overlay_boxes)}"
    cv2.putText(display, status,        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
    cv2.putText(display, f"T={THRESHOLD}",(10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(display, "Q = quit",    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    cv2.imshow("Live Recognition Test", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()