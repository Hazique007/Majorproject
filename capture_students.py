# capture_student.py
import cv2, os
import numpy as np
from ultralytics import YOLO
from db_config import get_db, verify_connection

verify_connection()
db           = get_db()
students_col = db["student_embeddings"]

model = YOLO("my_model.pt")

faculty_no = input("Enter faculty number: ")
name       = input("Enter student name (use _ for spaces): ")

existing = students_col.find_one({"faculty_no": faculty_no})
if existing:
    print(f"⚠ Student {existing['name']} already exists in DB!")
    choice = input("Re-capture and overwrite? (y/n): ")
    if choice.lower() != 'y':
        print("Exiting.")
        exit(0)

save_dir = f"student_faces/{faculty_no}_{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

# ── Tuned thresholds ──────────────────────────
count      = 0
MIN_PHOTOS = 8
FACE_CONF  = 0.40   # ← lowered from 0.75 (was rejecting valid faces)
MIN_SIZE   = 60     # ← lowered from 100 (was rejecting normal distances)
PADDING    = 30

print(f"\nCapturing for: {name} ({faculty_no})")
print("SPACE = capture | Q = quit")
print("Tip: Make sure your face is well-lit and centred\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    display    = frame.copy()
    h, w       = frame.shape[:2]

    # Run YOLO detection
    results    = model(frame, verbose=False)
    detections = results[0].boxes

    face_detected  = False
    best_face_crop = None
    best_conf      = 0
    best_box       = None

    for det in detections:
        conf = det.conf.item()

        # ── FIX: safe squeeze for any number of detections ──
        xyxy_np = det.xyxy.cpu().numpy()
        if xyxy_np.ndim == 1:
            xyxy = xyxy_np.astype(int)
        else:
            xyxy = xyxy_np.squeeze().astype(int)

        # Skip if still wrong shape
        if xyxy.shape != (4,):
            continue

        xmin, ymin, xmax, ymax = xyxy
        face_w = xmax - xmin
        face_h = ymax - ymin

        # Too small = too far away
        if face_w < MIN_SIZE or face_h < MIN_SIZE:
            cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 100, 255), 1)
            cv2.putText(display, f"Too far ({face_w}px)", (xmin, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            continue

        # Below confidence threshold
        if conf < FACE_CONF:
            cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 165, 255), 1)
            cv2.putText(display, f"Low conf ({int(conf*100)}%)", (xmin, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            continue

        # Valid face — keep best one
        if conf > best_conf:
            best_conf = conf
            best_box  = (xmin, ymin, xmax, ymax)

            x1 = max(0, xmin - PADDING)
            y1 = max(0, ymin - PADDING)
            x2 = min(w, xmax + PADDING)
            y2 = min(h, ymax + PADDING)
            best_face_crop = frame[y1:y2, x1:x2].copy()
            face_detected  = True

    # Draw best detection box
    if best_box is not None:
        xmin, ymin, xmax, ymax = best_box
        cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(display, f"Face: {int(best_conf*100)}%",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Status bar ────────────────────────────
    status_color = (0, 255, 0) if face_detected else (0, 0, 255)
    status_text  = "SPACE to capture" if face_detected else "No face — adjust position/lighting"
    cv2.putText(display, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(display, f"Saved: {count}/{MIN_PHOTOS}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display, f"{name} | {faculty_no}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Debug line — shows raw detection count so you know YOLO is working
    raw_count = len(detections)
    cv2.putText(display, f"YOLO raw detections: {raw_count}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Face preview thumbnail (top right)
    if best_face_crop is not None and best_face_crop.size > 0:
        try:
            preview = cv2.resize(best_face_crop, (90, 90))
            display[8:98, w-98:w-8] = preview
            cv2.rectangle(display, (w-98, 8), (w-8, 98), (0,255,0), 1)
        except:
            pass  # Skip preview if crop is malformed

    cv2.imshow(f"Capturing: {name}", display)
    key = cv2.waitKey(1)

    if key == ord(' '):
        if face_detected and best_face_crop is not None and best_face_crop.size > 0:
            count += 1
            path  = f"{save_dir}/{count}.jpg"
            cv2.imwrite(path, best_face_crop)
            print(f"✓ Photo {count} saved — conf: {best_conf:.2f} | "
                  f"size: {best_face_crop.shape[1]}x{best_face_crop.shape[0]}px")
            if count == MIN_PHOTOS:
                print(f"✅ Minimum {MIN_PHOTOS} photos reached! "
                      f"Take more for better accuracy or press Q.")
        else:
            print("⚠ No valid face detected — check lighting and distance")

    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*45}")
print(f"  Student  : {name.replace('_', ' ')}")
print(f"  Faculty# : {faculty_no}")
print(f"  Photos   : {count} saved → {save_dir}/")
print(f"  Status   : {'✅ Ready for enrollment!' if count >= 5 else '⚠ Too few photos, re-run!'}")
print(f"{'='*45}")
print("Next step → python enroll_students.py")