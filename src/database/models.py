from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    audio_files = relationship("AudioFile", back_populates="user")

class AudioFile(Base):
    __tablename__ = "audio_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)
    size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)
    format = Column(String(10), nullable=False)
    status = Column(String(20), default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)
    file_metadata = Column(Text, nullable=True)  # JSON as text for SQLite
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    # Relationships
    user = relationship("User", back_populates="audio_files")
    jobs = relationship("TranscriptionJob", back_populates="audio_file", cascade="all, delete-orphan")

class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"
    id = Column(Integer, primary_key=True, index=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"), nullable=False)
    status = Column(String(20), default="pending")
    progress = Column(Float, default=0.0)
    config = Column(Text, nullable=True)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    # Relationships
    audio_file = relationship("AudioFile", back_populates="jobs")
    transcription = relationship("Transcription", back_populates="job", uselist=False, cascade="all, delete-orphan")

class Transcription(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("transcription_jobs.id"), nullable=False)
    text = Column(Text, nullable=False)
    segments_json = Column(Text, nullable=True)  # JSON as text for SQLite
    confidence = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Quality analysis fields
    engine_used = Column(String(50), nullable=True)  # whisper, whisperx, faster-whisper
    model_size = Column(String(20), nullable=True)   # tiny, base, small, medium, large, large-v3
    vad_method = Column(String(20), nullable=True)   # none, simple, webrtcvad, silero
    language_detected = Column(String(10), nullable=True)
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    words_per_second = Column(Float, nullable=True)
    
    # VAD results
    vad_segments_count = Column(Integer, nullable=True)
    vad_total_speech_duration = Column(Float, nullable=True)
    vad_processing_time = Column(Float, nullable=True)
    
    # Speaker diarization
    speaker_segments_json = Column(Text, nullable=True)  # JSON as text for SQLite
    speaker_count = Column(Integer, nullable=True)
    
    # Performance metrics
    gpu_used = Column(String(100), nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    
    # Relationships
    job = relationship("TranscriptionJob", back_populates="transcription")
    segments = relationship("TranscriptionSegment", back_populates="transcription")

class TranscriptionSegment(Base):
    """Model for individual transcription segments with timestamps."""
    
    __tablename__ = "transcription_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    transcription_id = Column(Integer, ForeignKey("transcriptions.id"), nullable=False)
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)    # End time in seconds
    text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)   # Segment confidence score
    speaker_id = Column(String(50), nullable=True)  # Speaker identifier if diarization enabled
    
    # Relationships
    transcription = relationship("Transcription", back_populates="segments")
    
    def __repr__(self):
        return f"<TranscriptionSegment(id={self.id}, start={self.start_time}, end={self.end_time})>"


class Speaker(Base):
    """Model for speaker profiles (optional feature)."""
    
    __tablename__ = "speakers"
    
    id = Column(Integer, primary_key=True, index=True)
    speaker_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Speaker(id={self.id}, speaker_id='{self.speaker_id}', name='{self.name}')>"


class ProcessingJob(Base):
    """Model for tracking processing jobs."""
    
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"), nullable=False)
    job_type = Column(String(50), nullable=False)  # transcription, diarization, etc.
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)  # Progress percentage
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, status='{self.status}', progress={self.progress})>" 