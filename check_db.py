# check_db.py
from db_config import get_db, verify_connection

verify_connection()
db = get_db()
students = db["student_embeddings"]

all_students = list(students.find({}, {"name":1, "faculty_no":1, "embedding_dim":1, "photo_count":1}))
print(f"\n✅ Total students in DB: {len(all_students)}\n")
for s in all_students:
    print(f"  {s['faculty_no']} | {s['name']} | "
          f"dims: {s.get('embedding_dim','?')} | "
          f"photos: {s.get('photo_count','?')}")