import pytest
import tempfile
import os
import numpy as np
import soundfile as sf
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.database.models import Base, AudioFile, TranscriptionJob, Transcription, User
from src.database.init_db import get_db
from src.main import create_app
from src.config import settings


def create_test_audio_file(duration=2.0, sample_rate=16000, filename="test_audio.wav"):
    """Create a valid test audio file."""
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a 440 Hz sine wave
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    
    # Save as WAV file
    sf.write(temp_file.name, audio_data, sample_rate)
    
    return temp_file.name


@pytest.fixture
def sample_audio_files():
    """Create sample audio files for testing."""
    files = []
    
    # Create WAV file
    wav_file = create_test_audio_file(duration=2.0, filename="test.wav")
    files.append(wav_file)
    
    # Create MP3 file (we'll use WAV for now since MP3 encoding is complex)
    mp3_file = create_test_audio_file(duration=1.5, filename="test.mp3")
    files.append(mp3_file)
    
    yield files
    
    # Cleanup
    for file in files:
        if os.path.exists(file):
            os.unlink(file)


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def db_session():
    """Create a database session for testing."""
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session factory
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create session
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    # Use a temporary file-based SQLite DB
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        engine.dispose()
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture
def mock_upload_file():
    """Create a mock upload file for testing."""
    def _create_mock_file(filename="test.wav", content=b"test audio content"):
        mock_file = MagicMock()
        mock_file.filename = filename
        mock_file.file = MagicMock()
        mock_file.file.read.return_value = content
        mock_file.file.seek.return_value = None
        return mock_file
    
    return _create_mock_file


@pytest.fixture
def sample_user(db_session):
    """Create a sample user for testing."""
    user = User(name="Test User")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_audio_file(db_session, sample_user):
    """Create a sample audio file for testing."""
    audio_file = AudioFile(
        filename="test.wav",
        path="/path/to/test.wav",
        size=1024,
        duration=10.5,
        format="wav",
        user_id=sample_user.id
    )
    db_session.add(audio_file)
    db_session.commit()
    db_session.refresh(audio_file)
    return audio_file


@pytest.fixture
def sample_transcription_job(db_session, sample_audio_file):
    """Create a sample transcription job for testing."""
    job = TranscriptionJob(
        audio_file_id=sample_audio_file.id,
        status="completed",
        progress=100.0
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job


@pytest.fixture
def sample_transcription(db_session, sample_transcription_job):
    """Create a sample transcription for testing."""
    transcription = Transcription(
        job_id=sample_transcription_job.id,
        text="Hello world",
        confidence=0.95
    )
    db_session.add(transcription)
    db_session.commit()
    db_session.refresh(transcription)
    return transcription


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    class MockWhisperModel:
        def __call__(self, audio_path, **kwargs):
            return {
                "text": "Mock transcription result",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 3.0,
                        "text": "Mock transcription result",
                        "no_speech_prob": 0.1
                    }
                ],
                "language": "en"
            }
    
    return MockWhisperModel()


@pytest.fixture
def mock_diarizer():
    """Mock speaker diarizer for testing."""
    class MockDiarizer:
        def diarize(self, audio_path, max_speakers=10):
            return [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "speaker": "Speaker 1",
                    "confidence": 0.9
                }
            ]
    
    return MockDiarizer() 