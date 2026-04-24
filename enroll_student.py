# enroll_students.py
import os
import numpy as np
from deepface import DeepFace
from datetime import datetime
from db_config import get_db, verify_connection

verify_connection()
db           = get_db()
students_col = db["student_embeddings"]
students_col.create_index("faculty_no", unique=True)

STUDENT_DB = "student_faces/"
MODEL_NAME = "ArcFace"

def enroll_student(faculty_no, name, folder_path):
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))
    ]

    if not image_files:
        print(f"  ⚠ No images in {folder_path}")
        return False

    embeddings   = []
    failed_count = 0

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            result = DeepFace.represent(
                img_path          = img_path,
                model_name        = MODEL_NAME,
                enforce_detection = False,
                detector_backend  = "opencv"
            )
            emb = np.array(result[0]["embedding"])

            # ── Normalize each embedding ──
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            embeddings.append(emb)
            print(f"    ✓ {img_file}")
        except Exception as e:
            failed_count += 1
            print(f"    ✗ {img_file} → skipped ({e})")

    if not embeddings:
        print(f"  ❌ No valid embeddings for {name}")
        return False

    # Average then re-normalize
    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)

    students_col.update_one(
        {"faculty_no": faculty_no},
        {"$set": {
            "name"          : name,
            "faculty_no"    : faculty_no,
            "embedding"     : avg.tolist(),
            "embedding_dim" : len(avg),
            "model"         : MODEL_NAME,
            "photo_count"   : len(embeddings),
            "failed_photos" : failed_count,
            "enrolled_at"   : datetime.utcnow(),
            "last_updated"  : datetime.utcnow()
        }},
        upsert=True
    )
    return True

def main():
    folders = [
        f for f in os.listdir(STUDENT_DB)
        if os.path.isdir(os.path.join(STUDENT_DB, f))
    ]
    print(f"Found {len(folders)} folders\n{'='*50}")

    success_count = 0
    failed_list   = []

    for folder in folders:
        if "_" not in folder:
            print(f"⚠ Skipping '{folder}'")
            continue

        parts      = folder.split("_", 1)
        faculty_no = parts[0].strip()
        name       = parts[1].replace("_"," ").strip()
        folder_path = os.path.join(STUDENT_DB, folder)

        print(f"\n👤 {name} | {faculty_no}")
        if enroll_student(faculty_no, name, folder_path):
            success_count += 1
            print(f"  ✅ Enrolled")
        else:
            failed_list.append(f"{faculty_no} - {name}")

    print(f"\n{'='*50}")
    print(f"  Enrolled : {success_count}/{len(folders)}")
    print(f"  In Atlas : {students_col.count_documents({})} total")
    if failed_list:
        print("  Failed:")
        for f in failed_list:
            print(f"    ✗ {f}")
    print(f"{'='*50}")
    print("✅ Next → python live_testing.py")

if __name__ == "__main__":
    main()