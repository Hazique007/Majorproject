# db_config.py
from pymongo import MongoClient

ATLAS_URI = "mongodb+srv://legendhaz2303_db_user:98LI7Zop1vEhueij@majorproject.seljzfk.mongodb.net/?appName=Majorproject"

def get_db():
    client = MongoClient(ATLAS_URI)
    db = client["attendance_system"]
    return db

def verify_connection():
    try:
        client = MongoClient(ATLAS_URI)
        client.admin.command("ping")
        print("✅ Connected to MongoDB Atlas — Majorproject cluster")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("→ Check your internet connection or Atlas IP whitelist")
        exit(1)