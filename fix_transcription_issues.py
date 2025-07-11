#!/usr/bin/env python3
"""
Fix transcription issues for long audio files.
The problem is that faster-whisper is not transcribing the full audio file.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine
from config import settings
import librosa

def test_transcription_with_different_settings():
    """Test transcription with different settings to find the best approach."""
    
    audio_path = 'uploads/c942fd8d-c949-43eb-8246-5abe431073ec.m4a'
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Test different configurations
    configs = [
        {
            "name": "Standard Whisper (base)",
            "engine": "whisper",
            "model_size": "base",
            "device": "cpu",
            "vad_method": "simple",
            "enable_speaker_diarization": False
        },
        {
            "name": "Standard Whisper (large-v3)",
            "engine": "whisper", 
            "model_size": "large-v3",
            "device": "cpu",
            "vad_method": "simple",
            "enable_speaker_diarization": False
        },
        {
            "name": "Faster Whisper (base) - No VAD",
            "engine": "faster-whisper",
            "model_size": "base",
            "device": "cpu",
            "vad_method": "simple",
            "enable_speaker_diarization": False
        },
        {
            "name": "Faster Whisper (base) - With VAD",
            "engine": "faster-whisper",
            "model_size": "base", 
            "device": "cpu",
            "vad_method": "simple",
            "enable_speaker_diarization": False
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Create engine with specific settings
            engine = TranscriptionEngine(
                model_size=config['model_size'],
                device=config['device'],
                engine=config['engine'],
                vad_method=config['vad_method'],
                enable_speaker_diarization=config['enable_speaker_diarization'],
                compute_type="float32" if config['device'] == "cpu" else "float16"
            )
            
            # Transcribe
            result = engine.transcribe(audio_path, language="en")
            
            # Analyze results
            text = result.get('text', '')
            segments = result.get('segments', [])
            processing_time = result.get('processing_time', 0)
            
            print(f"✅ Processing time: {processing_time:.2f}s")
            print(f"📝 Text length: {len(text)} characters")
            print(f"📝 Word count: {len(text.split())} words")
            print(f"🎯 Segments: {len(segments)}")
            
            if segments:
                last_segment = segments[-1]
                print(f"⏰ Last segment ends at: {last_segment.get('end', 0):.2f}s")
                print(f"📊 Coverage: {(last_segment.get('end', 0) / duration) * 100:.1f}%")
            
            # Check if transcription seems complete
            if len(text) < 1000:
                print("⚠️  WARNING: Very short transcription!")
            elif text.endswith('...') or text.endswith('..'):
                print("⚠️  WARNING: Transcription ends with ellipsis!")
            else:
                print("✅ Transcription appears complete")
            
            # Show first and last 200 characters
            print(f"\nFirst 200 chars: {text[:200]}...")
            print(f"Last 200 chars: ...{text[-200:]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def update_configuration():
    """Update the configuration to fix transcription issues."""
    
    print("🔧 Updating configuration to fix transcription issues...")
    
    # Update src/config.py
    config_content = '''import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
import sys
import logging
from loguru import logger


class Settings(BaseSettings):
    """Application settings for local transcription platform."""
    
    # Database
    database_url: str = "sqlite:///./transcription.db"
    
    # File storage
    upload_dir: str = "./uploads"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    
    # Whisper configuration - Updated for better long audio handling
    whisper_model: str = "large-v3"  # Use large-v3 for better accuracy on long files
    language: str = "en"
    device: str = "cpu"  # Use CPU for less resource usage
    
    # Transcription engine options - Use standard Whisper for long files
    transcription_engine: str = "whisper"  # Changed from "faster-whisper" to "whisper"
    
    # VAD (Voice Activity Detection) options - Disable for long files
    vad_method: str = "simple"  # Options: "simple", "webrtcvad", "silero"
    enable_vad: bool = False  # Disabled VAD for long audio files
    
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
    port: int = 8000
    debug: bool = True
    
    # Audio processing - Increased limits for long files
    max_file_size: str = "500MB"  # Increased from 100MB
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
    
    # Performance - Increased timeout for long files
    max_concurrent_jobs: int = 3  # Reduced for memory management
    job_timeout: int = 3600  # Increased to 1 hour for long files
    
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
'''
    
    with open('src/config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration updated!")
    print("   - Changed engine from faster-whisper to whisper")
    print("   - Changed model from base to large-v3")
    print("   - Disabled VAD for long audio files")
    print("   - Increased max file size to 500MB")
    print("   - Increased job timeout to 1 hour")
    print("   - Reduced concurrent jobs to 3")

if __name__ == "__main__":
    print("🔧 Transcription Issue Fixer")
    print("=" * 50)
    
    # First test current settings
    print("🧪 Testing current transcription settings...")
    test_transcription_with_different_settings()
    
    # Update configuration
    print("\n" + "="*60)
    update_configuration()
    
    print("\n" + "="*60)
    print("📋 Next steps:")
    print("1. Restart the transcription service")
    print("2. Try transcribing the same file again")
    print("3. The new settings should handle long audio files better")
    print("4. If issues persist, consider using chunked processing") 