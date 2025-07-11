import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
import sys
import logging
from loguru import logger
import warnings


class Settings(BaseSettings):
    """Application settings for local transcription platform."""
    
    # Database
    database_url: str = "sqlite:///./transcription.db"
    
    # File storage
    upload_dir: str = "./uploads"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    
    # Whisper configuration - Optimized for CUDA performance
    whisper_model: str = "medium"  # Balanced speed/accuracy for CUDA
    language: str = "en"
    device: str = "cuda"  # Use CUDA for best performance
    
    # Transcription engine options
    transcription_engine: str = "faster-whisper"  # Best CUDA performance
    default_model_size: str = "medium"  # Balanced speed/accuracy for CUDA
    
    # VAD (Voice Activity Detection) options
    vad_method: str = "none"  # Disable VAD for speed
    enable_vad: bool = False  # Disable VAD for speed
    
    # Audio preprocessing settings for better transcription
    enable_audio_preprocessing: bool = False  # Disable for speed
    enable_denoising: bool = False  # Disable for speed
    enable_normalization: bool = False  # Disable for speed
    
    # Audio preprocessing parameters
    highpass_freq: int = 200  # High-pass filter frequency
    lowpass_freq: int = 3000  # Low-pass filter frequency
    target_sample_rate: int = 16000
    normalize_audio: bool = True
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Audio processing - Fixed duplicate definitions
    max_file_size_str: str = "500MB"  # String representation for display
    max_file_size_bytes: int = 500 * 1024 * 1024  # 500MB in bytes
    supported_formats: str = "wav,mp3,m4a,flac"
    sample_rate: int = 16000
    
    # File upload settings
    upload_directory: str = "./uploads"
    allowed_extensions: List[str] = ["mp3", "wav", "m4a", "flac", "ogg"]
    
    # Optional: Speaker diarization
    enable_speaker_diarization: bool = False
    pyannote_model: str = "pyannote/speaker-diarization@2.1"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    suppress_verbose_logs: bool = True
    show_sql_queries: bool = False
    show_model_loading_details: bool = False
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    allowed_hosts: str = "localhost,127.0.0.1"
    
    # Performance
    max_concurrent_jobs: int = 5
    job_timeout: int = 1800
    
    class Config:
        env_file = ".env.local"
        case_sensitive = False
    
    @property
    def supported_formats_list(self) -> List[str]:
        """Get list of supported audio formats."""
        return [fmt.strip() for fmt in self.supported_formats.split(",")]
    
    @property
    def allowed_hosts_list(self) -> List[str]:
        """Get list of allowed hosts."""
        return [host.strip() for host in self.allowed_hosts.split(",")]
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            Path(self.upload_dir),
            Path(self.models_dir),
            Path(self.logs_dir),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Configure logging
logger.remove()  # Remove default handler

# Completely suppress SQLAlchemy verbosity - do this BEFORE any other imports
import logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.orm").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)

# Also suppress other noisy loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Set loguru to show only important messages
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
    colorize=False,
    filter=lambda record: record["level"].no >= 20  # Only INFO and above
)
logger.add(
    settings.log_file,
    level=settings.log_level,
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    colorize=False
)

# Suppress verbose logs
def configure_logging():
    """Configure logging to show only essential information."""
    
    # Suppress SQLAlchemy verbose logging
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
    
    # Suppress other verbose modules
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('librosa').setLevel(logging.WARNING)
    
    # Suppress SpeechBrain deprecation warning
    warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="speechbrain")
    
    # Configure loguru for clean output
    from loguru import logger
    logger.remove()  # Remove default handler
    
    # Add custom handler with minimal format
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days"
    )
    
    # Add console handler with minimal output
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>\n",
        level="INFO",
        filter=lambda record: record["level"].no >= 20  # Only INFO and above
    ) 