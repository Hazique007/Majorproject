# score_distribution.py
import os, numpy as np
from deepface import DeepFace
from db_config import get_db, verify_connection

verify_connection()
db = get_db()
all_students = list(db["student_embeddings"].find({}, {"name":1, "faculty_no":1, "embedding":1}))

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / norm if norm > 0 else 0.0

# Test EVERY saved photo of EVERY student
for student in all_students:
    folder = f"student_faces/{student['faculty_no']}_{student['name'].replace(' ', '_')}"
    if not os.path.exists(folder):
        continue

    scores = []
    for img_file in os.listdir(folder):
        if not img_file.lower().endswith((".jpg",".jpeg",".png")):
            continue
        img_path = os.path.join(folder, img_file)
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="opencv"
            )
            emb = result[0]["embedding"]
            score = cosine_similarity(emb, student["embedding"])
            scores.append(score)
        except:
            pass

    if scores:
        print(f"\n{student['name']} ({student['faculty_no']})")
        print(f"  Min : {min(scores):.4f}")
        print(f"  Max : {max(scores):.4f}")
        print(f"  Avg : {np.mean(scores):.4f}  ← this is your natural threshold")