import pytest
import tempfile
import os
from datetime import datetime
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, AudioFile, TranscriptionJob, Transcription, User
from src.database.init_db import init_database


class TestDatabase:
    """Test cases for database operations."""
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Create temporary database
        test_db_path = tempfile.mktemp(suffix=".db")
        
        try:
            # Create engine
            engine = create_engine(f"sqlite:///{test_db_path}")
            
            # Initialize database
            from src.database.init_db import create_tables
            create_tables(engine)
            
            # Check if tables exist
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            expected_tables = [
                "users", "audio_files", "transcription_jobs", 
                "transcriptions", "transcription_segments", 
                "speakers", "processing_jobs"
            ]
            
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"
                
        finally:
            if os.path.exists(test_db_path):
                os.unlink(test_db_path)
    
    def test_audio_file_creation(self, db_session):
        """Test creating audio file records."""
        # Create audio file
        audio_file = AudioFile(
            filename="test.wav",
            path="/path/to/test.wav",
            size=1024,
            duration=10.5,
            format="wav",
            status="uploaded"
        )
        
        db_session.add(audio_file)
        db_session.commit()
        db_session.refresh(audio_file)
        
        # Check that ID was assigned
        assert audio_file.id is not None
        assert audio_file.filename == "test.wav"
        assert audio_file.duration == 10.5
    
    def test_transcription_job_creation(self, db_session):
        """Test creating transcription job records."""
        # Create audio file first
        audio_file = AudioFile(
            filename="test.wav",
            path="/path/to/test.wav",
            size=1024,
            format="wav"
        )
        db_session.add(audio_file)
        db_session.commit()
        
        # Create transcription job
        job = TranscriptionJob(
            audio_file_id=audio_file.id,
            status="pending",
            progress=0.0
        )
        
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        
        # Check that ID was assigned
        assert job.id is not None
        assert job.audio_file_id == audio_file.id
        assert job.status == "pending"
    
    def test_transcription_creation(self, db_session):
        """Test creating transcription records."""
        # Create audio file and job
        audio_file = AudioFile(
            filename="test.wav",
            path="/path/to/test.wav",
            size=1024,
            format="wav"
        )
        db_session.add(audio_file)
        db_session.commit()
        
        job = TranscriptionJob(
            audio_file_id=audio_file.id,
            status="completed"
        )
        db_session.add(job)
        db_session.commit()
        
        # Create transcription
        transcription = Transcription(
            job_id=job.id,
            text="Hello world",
            confidence=0.95
        )
        
        db_session.add(transcription)
        db_session.commit()
        db_session.refresh(transcription)
        
        # Check that ID was assigned
        assert transcription.id is not None
        assert transcription.text == "Hello world"
        assert transcription.confidence == 0.95
    
    def test_relationships(self, db_session):
        """Test database relationships."""
        # Create user
        user = User(name="Test User")
        db_session.add(user)
        db_session.commit()
        
        # Create audio file with user
        audio_file = AudioFile(
            filename="test.wav",
            path="/path/to/test.wav",
            size=1024,
            format="wav",
            user_id=user.id
        )
        db_session.add(audio_file)
        db_session.commit()
        
        # Test relationship
        assert audio_file.user.name == "Test User"
        assert len(user.audio_files) == 1
        assert user.audio_files[0].filename == "test.wav"
    
    def test_query_operations(self, db_session):
        """Test basic query operations."""
        # Create multiple audio files
        for i in range(5):
            audio_file = AudioFile(
                filename=f"test_{i}.wav",
                path=f"/path/to/test_{i}.wav",
                size=1024,
                format="wav"
            )
            db_session.add(audio_file)
        
        db_session.commit()
        
        # Test queries
        all_files = db_session.query(AudioFile).all()
        assert len(all_files) == 5
        
        wav_files = db_session.query(AudioFile).filter(AudioFile.format == "wav").all()
        assert len(wav_files) == 5
        
        first_file = db_session.query(AudioFile).first()
        assert first_file is not None
        assert first_file.filename.startswith("test_")
    
    def test_transcription_with_segments(self, db_session):
        """Test transcription with segments."""
        # Create audio file and job
        audio_file = AudioFile(
            filename="test.wav",
            path="/path/to/test.wav",
            size=1024,
            format="wav"
        )
        db_session.add(audio_file)
        db_session.commit()
        
        job = TranscriptionJob(
            audio_file_id=audio_file.id,
            status="completed"
        )
        db_session.add(job)
        db_session.commit()
        
        # Create transcription
        transcription = Transcription(
            job_id=job.id,
            text="Hello world",
            confidence=0.95
        )
        db_session.add(transcription)
        db_session.commit()
        
        # Create segments
        from src.database.models import TranscriptionSegment
        
        segment1 = TranscriptionSegment(
            transcription_id=transcription.id,
            start_time=0.0,
            end_time=2.0,
            text="Hello",
            confidence=0.9,
            speaker_id="SPEAKER_00"
        )
        
        segment2 = TranscriptionSegment(
            transcription_id=transcription.id,
            start_time=2.0,
            end_time=4.0,
            text="world",
            confidence=0.95,
            speaker_id="SPEAKER_00"
        )
        
        db_session.add_all([segment1, segment2])
        db_session.commit()
        
        # Test relationships
        assert len(transcription.segments_rel) == 2
        assert transcription.segments_rel[0].text == "Hello"
        assert transcription.segments_rel[1].text == "world"
    
    def test_data_integrity(self, db_session):
        """Test data integrity constraints."""
        # Test that we can't create audio file without required fields
        with pytest.raises(Exception):
            audio_file = AudioFile()  # Missing required fields
            db_session.add(audio_file)
            db_session.commit()
    
    def test_cascade_deletion(self, db_session):
        """Test cascade deletion."""
        # Create audio file
        audio_file = AudioFile(
            filename="test.wav",
            path="/path/to/test.wav",
            size=1024,
            format="wav"
        )
        db_session.add(audio_file)
        db_session.commit()
        
        # Create job
        job = TranscriptionJob(
            audio_file_id=audio_file.id,
            status="pending"
        )
        db_session.add(job)
        db_session.commit()
        
        # Delete audio file
        db_session.delete(audio_file)
        db_session.commit()
        
        # Check if job is also deleted
        remaining_job = db_session.query(TranscriptionJob).filter_by(id=job.id).first()
        assert remaining_job is None
    
    def test_query_performance(self, db_session):
        """Test query performance with multiple records."""
        # Create multiple audio files
        for i in range(100):
            audio_file = AudioFile(
                filename=f"test_{i}.wav",
                path=f"/path/to/test_{i}.wav",
                size=1024,
                format="wav"
            )
            db_session.add(audio_file)
        
        db_session.commit()
        
        # Test query performance
        import time
        start_time = time.time()
        
        files = db_session.query(AudioFile).all()
        
        query_time = time.time() - start_time
        
        assert len(files) == 100
        assert query_time < 1.0  # Should be fast 