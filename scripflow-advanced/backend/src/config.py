import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for local transcription platform."""
    
    # Database
    database_url: str = "sqlite:///./transcription.db"
    
    # File storage
    upload_dir: str = "./uploads"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    
    # Whisper configuration - Updated for Indian accents
    whisper_model: str = "base"  # Changed from "base" to "large-v3" for better accuracy
    language: str = "en"
    device: str = "cpu"  # Use CPU for less resource usage
    
    # Transcription engine options
    transcription_engine: str = "whisper"  # Options: "whisper", "whisperx"
    
    # VAD (Voice Activity Detection) options
    vad_method: str = "simple"  # Options: "simple", "webrtcvad", "silero"
    enable_vad: bool = True  # Voice Activity Detection
    
    # Audio preprocessing settings for better transcription
    enable_audio_preprocessing: bool = True
    enable_denoising: bool = True
    enable_normalization: bool = True
    
    # Audio preprocessing parameters
    highpass_freq: int = 200  # High-pass filter frequency
    lowpass_freq: int = 3000  # Low-pass filter frequency
    target_sample_rate: int = 16000
    normalize_audio: bool = True
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8010
    debug: bool = True
    
    # Audio processing
    max_file_size: str = "100MB"
    supported_formats: str = "wav,mp3,m4a,flac"
    sample_rate: int = 16000
    
    # Optional: Speaker diarization
    enable_speaker_diarization: bool = False
    pyannote_model: str = "pyannote/speaker-diarization@2.1"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
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
    def max_file_size_bytes(self) -> int:
        """Convert max file size string to bytes."""
        size_str = self.max_file_size.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
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