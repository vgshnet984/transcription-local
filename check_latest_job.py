#!/usr/bin/env python3

import sqlite3

# Connect to the database
conn = sqlite3.connect('transcription.db')
cursor = conn.cursor()

# Get the latest job
cursor.execute("SELECT id, status, progress, audio_file_id FROM transcription_jobs ORDER BY id DESC LIMIT 5")
jobs = cursor.fetchall()

print("Latest jobs:")
for job in jobs:
    print(f"Job ID: {job[0]}, Status: {job[1]}, Progress: {job[2]}, Audio File ID: {job[3]}")

# Get the latest transcription
cursor.execute("SELECT id, job_id, text FROM transcriptions ORDER BY id DESC LIMIT 5")
transcriptions = cursor.fetchall()

print("\nLatest transcriptions:")
for trans in transcriptions:
    print(f"Transcription ID: {trans[0]}, Job ID: {trans[1]}, Text preview: {trans[2][:50]}...")

conn.close() 