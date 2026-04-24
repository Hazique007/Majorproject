# capture_student.py
import cv2, os
import numpy as np
from ultralytics import YOLO
from db_config import get_db, verify_connection

verify_connection()
db           = get_db()
students_col = db["student_embeddings"]

model = YOLO("my_model.pt")

# ── Camera config ─────────────────────────────
CAMERA_IP = "192.168.1.64"
USERNAME  = "admin"
PASSWORD  = "JABIN002@yunus"
RTSP_URL  = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/102"

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

# Clear old photos if overwriting
for f in os.listdir(save_dir):
    if f.lower().endswith((".jpg",".jpeg",".png")):
        os.remove(os.path.join(save_dir, f))
print(f"🗑 Old photos cleared\n")

# ── Connect to CCTV ───────────────────────────
print("Connecting to CCTV camera...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("❌ CCTV failed — trying webcam fallback...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available")
        exit(1)
    print("✅ Using webcam")
else:
    print("✅ CCTV connected\n")

# ── Thresholds ────────────────────────────────
count      = 0
MIN_PHOTOS = 15   # more photos for better CCTV accuracy
FACE_CONF  = 0.30  # lower for CCTV angle
MIN_SIZE   = 40    # lower for CCTV distance
PADDING    = 40

print(f"Capturing for: {name} ({faculty_no})")
print("SPACE = capture | Q = quit")
print("Tip: Stand at normal classroom distance from camera\n")

while True:
    # Drain buffer for latest frame
    frame = None
    for _ in range(4):
        ret, f = cap.read()
        if ret:
            frame = f

    if frame is None:
        print("⚠ Frame failed — reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    display = frame.copy()
    h, w    = frame.shape[:2]

    results    = model(frame, verbose=False)
    detections = results[0].boxes

    face_detected  = False
    best_face_crop = None
    best_conf      = 0
    best_box       = None

    for det in detections:
        conf    = det.conf.item()
        xyxy_np = det.xyxy.cpu().numpy()
        if xyxy_np.ndim == 1:
            xyxy = xyxy_np.astype(int)
        else:
            xyxy = xyxy_np.squeeze().astype(int)

        if xyxy.shape != (4,):
            continue

        xmin, ymin, xmax, ymax = xyxy
        face_w = xmax - xmin
        face_h = ymax - ymin

        if face_w < MIN_SIZE or face_h < MIN_SIZE:
            cv2.rectangle(display, (xmin,ymin), (xmax,ymax), (0,100,255), 1)
            cv2.putText(display, f"Too far ({face_w}px)", (xmin, ymin-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 1)
            continue

        if conf < FACE_CONF:
            cv2.rectangle(display, (xmin,ymin), (xmax,ymax), (0,165,255), 1)
            cv2.putText(display, f"Low conf ({int(conf*100)}%)", (xmin, ymin-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)
            continue

        if conf > best_conf:
            best_conf = conf
            best_box  = (xmin, ymin, xmax, ymax)
            x1 = max(0, xmin - PADDING)
            y1 = max(0, ymin - PADDING)
            x2 = min(w, xmax + PADDING)
            y2 = min(h, ymax + PADDING)
            best_face_crop = frame[y1:y2, x1:x2].copy()
            face_detected  = True

    if best_box is not None:
        xmin, ymin, xmax, ymax = best_box
        cv2.rectangle(display, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
        cv2.putText(display, f"Face: {int(best_conf*100)}%", (xmin, ymin-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    status_color = (0,255,0) if face_detected else (0,0,255)
    status_text  = "SPACE to capture" if face_detected else "No face — adjust position"
    cv2.putText(display, status_text,          (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(display, f"Saved: {count}/{MIN_PHOTOS}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(display, f"{name} | {faculty_no}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(display, f"YOLO detections: {len(detections)}", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    # Face preview thumbnail
    if best_face_crop is not None and best_face_crop.size > 0:
        try:
            preview = cv2.resize(best_face_crop, (90,90))
            display[8:98, w-98:w-8] = preview
            cv2.rectangle(display, (w-98,8), (w-8,98), (0,255,0), 1)
        except:
            pass

    # Resize for display
    display = cv2.resize(display, (800, 450))
    cv2.imshow(f"CCTV Capture: {name}", display)
    key = cv2.waitKey(1)

    if key == ord(' '):
        if face_detected and best_face_crop is not None and best_face_crop.size > 0:
            count += 1
            path  = f"{save_dir}/{count}.jpg"
            cv2.imwrite(path, best_face_crop)
            print(f"✓ Photo {count} saved — conf: {best_conf:.2f} | "
                  f"size: {best_face_crop.shape[1]}x{best_face_crop.shape[0]}px")
            if count == MIN_PHOTOS:
                print(f"✅ Minimum {MIN_PHOTOS} photos reached! Take more or press Q.")
        else:
            print("⚠ No valid face — adjust position/lighting")

    elif key in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*45}")
print(f"  Student  : {name.replace('_',' ')}")
print(f"  Faculty# : {faculty_no}")
print(f"  Photos   : {count} saved → {save_dir}/")
print(f"  Status   : {'✅ Ready!' if count >= 10 else '⚠ Too few photos, re-run!'}")
print(f"{'='*45}")
print("Next → python enroll_students.py")