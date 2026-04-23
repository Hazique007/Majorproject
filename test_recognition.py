# test_recognition.py
import cv2
import numpy as np
from deepface import DeepFace
from db_config import get_db, verify_connection

verify_connection()
db = get_db()
students_col = db["student_embeddings"]
all_students = list(students_col.find({}, {"name":1, "faculty_no":1, "embedding":1}))
print(f"Loaded {len(all_students)} students")

THRESHOLD = 0.55

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / norm if norm > 0 else 0.0

# ✏️ Change this to any saved student photo
TEST_IMAGE = "student_faces/22ELB284_Mohammad_Hazique_Khan/1.jpg"

result = DeepFace.represent(
    img_path=TEST_IMAGE,
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="opencv"
)
query_emb = result[0]["embedding"]

best_name, best_score = "Unknown", -1
for student in all_students:
    score = cosine_similarity(query_emb, student["embedding"])
    if score > best_score:
        best_score = score
        best_name  = student["name"]

print(f"\n🔍 Result: {best_name}")
print(f"   Score : {best_score:.4f}  ({'✅ MATCH' if best_score >= THRESHOLD else '❌ NO MATCH'})")
print(f"   Threshold: {THRESHOLD}")