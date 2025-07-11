import librosa
import sqlite3
import os

def check_audio_and_transcription():
    # Check audio file
    audio_path = 'uploads/c942fd8d-c949-43eb-8246-5abe431073ec.m4a'
    
    if os.path.exists(audio_path):
        try:
            duration = librosa.get_duration(path=audio_path)
            print(f"Audio file: {audio_path}")
            print(f"Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"File size: {os.path.getsize(audio_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"Error reading audio file: {e}")
    else:
        print(f"Audio file not found: {audio_path}")
    
    # Check transcription timing
    conn = sqlite3.connect('transcription.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT t.processing_time, tj.status, tj.progress, af.filename
        FROM transcriptions t 
        JOIN transcription_jobs tj ON t.job_id = tj.id 
        JOIN audio_files af ON tj.audio_file_id = af.id 
        WHERE t.job_id = 95
    ''')
    
    result = cursor.fetchone()
    if result:
        processing_time, status, progress, filename = result
        print(f"\nTranscription job info:")
        print(f"Status: {status}")
        print(f"Progress: {progress}%")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Calculate processing speed
        if duration and processing_time:
            speed = duration / processing_time
            print(f"Processing speed: {speed:.2f}x real-time")
    
    conn.close()

if __name__ == "__main__":
    check_audio_and_transcription() 