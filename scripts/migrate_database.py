#!/usr/bin/env python3
"""
Database migration script to add quality analysis fields to Transcription table.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.init_db import create_database_engine
from sqlalchemy import text

def migrate_database():
    """Add new quality analysis fields to the transcriptions table."""
    print("🔄 Starting database migration...")
    
    try:
        # Create database engine
        engine = create_database_engine()
        
        # Check if new columns already exist
        with engine.connect() as conn:
            # Get existing columns
            result = conn.execute(text("PRAGMA table_info(transcriptions)"))
            existing_columns = [row[1] for row in result.fetchall()]
            
            print(f"📋 Existing columns: {existing_columns}")
            
            # New columns to add
            new_columns = [
                ("engine_used", "VARCHAR(50)"),
                ("model_size", "VARCHAR(20)"),
                ("vad_method", "VARCHAR(20)"),
                ("language_detected", "VARCHAR(10)"),
                ("word_count", "INTEGER"),
                ("character_count", "INTEGER"),
                ("words_per_second", "FLOAT"),
                ("vad_segments_count", "INTEGER"),
                ("vad_total_speech_duration", "FLOAT"),
                ("vad_processing_time", "FLOAT"),
                ("speaker_segments_json", "TEXT"),
                ("speaker_count", "INTEGER"),
                ("gpu_used", "VARCHAR(100)"),
                ("memory_usage_mb", "FLOAT")
            ]
            
            # Add missing columns
            for column_name, column_type in new_columns:
                if column_name not in existing_columns:
                    print(f"➕ Adding column: {column_name}")
                    conn.execute(text(f"ALTER TABLE transcriptions ADD COLUMN {column_name} {column_type}"))
                    conn.commit()
                else:
                    print(f"✅ Column already exists: {column_name}")
            
            print("✅ Database migration completed successfully!")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    migrate_database() 