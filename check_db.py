import sqlite3

conn = sqlite3.connect('transcription.db')
cursor = conn.cursor()
cursor.execute('SELECT id, filename, path FROM audio_files WHERE id = 63')
result = cursor.fetchone()
print("Audio file record:", result)
conn.close() 