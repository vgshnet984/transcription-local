import sqlite3

def check_latest_transcription():
    conn = sqlite3.connect('transcription.db')
    cursor = conn.cursor()
    
    # Get the latest transcription
    cursor.execute('''
        SELECT t.text, t.job_id, af.filename, af.path 
        FROM transcriptions t 
        JOIN transcription_jobs tj ON t.job_id = tj.id 
        JOIN audio_files af ON tj.audio_file_id = af.id 
        ORDER BY t.id DESC 
        LIMIT 1
    ''')
    
    result = cursor.fetchone()
    if result:
        text, job_id, filename, path = result
        print(f"Latest transcription (Job ID: {job_id})")
        print(f"Audio file: {filename}")
        print(f"File path: {path}")
        print(f"Text length: {len(text)} characters")
        print(f"Text length: {len(text.split())} words")
        print("\nFirst 500 characters:")
        print("-" * 50)
        print(text[:500])
        print("\nLast 500 characters:")
        print("-" * 50)
        print(text[-500:])
        
        # Check if it seems incomplete
        if len(text) < 1000:  # Very short transcription
            print("\n⚠️  WARNING: Transcription seems very short!")
        elif text.endswith('...') or text.endswith('..') or text.endswith('.'):
            print("\n⚠️  WARNING: Transcription ends with ellipsis - may be incomplete!")
        else:
            print("\n✅ Transcription appears complete")
    else:
        print("No transcriptions found")
    
    conn.close()

if __name__ == "__main__":
    check_latest_transcription() 