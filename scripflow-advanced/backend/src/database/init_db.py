from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

from src.config import settings
from src.database.models import Base, User, AudioFile, TranscriptionJob, Transcription


def get_database_url() -> str:
    """Get database URL from settings."""
    return settings.database_url


def create_database_engine():
    """Create SQLAlchemy engine."""
    database_url = get_database_url()
    
    # For SQLite, ensure the database file directory exists
    if database_url.startswith("sqlite:///"):
        import os
        db_path = database_url.replace("sqlite:///", "")
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    engine = create_engine(
        database_url,
        echo=settings.debug,  # Log SQL queries in debug mode
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
    )
    
    return engine


def create_tables(engine):
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_session_factory(engine):
    """Create session factory for database operations."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize the database with tables."""
    try:
        engine = create_database_engine()
        create_tables(engine)
        logger.info("Database initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


# Global engine and session factory
engine = None
SessionLocal = None


def get_db():
    """Get database session."""
    global engine, SessionLocal
    
    if engine is None:
        engine = create_database_engine()
        SessionLocal = get_session_factory(engine)
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    engine = create_engine(settings.database_url, echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    # Sample data
    if not session.query(User).first():
        user = User(name="testuser")
        session.add(user)
        session.commit()
        audio = AudioFile(filename="sample.wav", path="/uploads/sample.wav", size=123456, duration=12.3, format="wav", user_id=user.id)
        session.add(audio)
        session.commit()
        job = TranscriptionJob(audio_file_id=audio.id, status="completed", progress=100.0)
        session.add(job)
        session.commit()
        transcription = Transcription(job_id=job.id, text="Hello world", confidence=0.99)
        session.add(transcription)
        session.commit()
    session.close()
    print("Database initialized and sample data inserted.")


if __name__ == "__main__":
    init_db() 