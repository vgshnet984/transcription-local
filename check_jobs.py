from src.database.init_db import SessionLocal
from src.database.models import TranscriptionJob

db = SessionLocal()
jobs = db.query(TranscriptionJob).order_by(TranscriptionJob.id.desc()).limit(10).all()

print("Latest 10 transcription jobs:")
for job in jobs:
    print(f"Job {job.id}: {job.status} - {job.progress}% - Audio file: {job.audio_file_id}")

db.close() 