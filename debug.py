# debug_match.py
import cv2
import numpy as np
from deepface import DeepFace
from db_config import get_db, verify_connection

verify_connection()
db = get_db()
all_students = list(db["student_embeddings"].find({}, {"name":1,"faculty_no":1,"embedding":1}))

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / norm if norm > 0 else 0.0

# Test with a saved student photo
TEST_IMAGE = "student_faces/22ELB283_Saad_Khan/1.jpg"

result = DeepFace.represent(
    img_path=TEST_IMAGE,
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="opencv"
)
query_emb = result[0]["embedding"]

print("\n📊 Scores against all students:")
for student in all_students:
    score = cosine_similarity(query_emb, student["embedding"])
    print(f"  {student['name']}: {score:.4f}")