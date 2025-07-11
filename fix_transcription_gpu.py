#!/usr/bin/env python3
"""
Fix transcription issues for long audio files using GPU acceleration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine
import librosa
import torch

def check_gpu_availability():
    """Check if GPU is available and which one."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU available: {gpu_name}")
        print(f"📊 GPU count: {gpu_count}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ No GPU available, will use CPU")
        return False

def test_transcription_with_gpu():
    """Test transcription with GPU acceleration."""
    
    audio_path = 'uploads/c942fd8d-c949-43eb-8246-5abe431073ec.m4a'
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Check GPU
    gpu_available = check_gpu_availability()
    device = "cuda" if gpu_available else "cpu"
    
    # Test different GPU configurations
    configs = [
        {
            "name": "Whisper Large-v3 (GPU)",
            "engine": "whisper",
            "model_size": "large-v3",
            "device": device,
            "vad_method": "simple",
            "enable_speaker_diarization": False
        },
        {
            "name": "WhisperX Large-v3 (GPU)",
            "engine": "whisperx",
            "model_size": "large-v3",
            "device": device,
            "vad_method": "simple",
            "enable_speaker_diarization": False
        },
        {
            "name": "Faster Whisper Large-v3 (GPU)",
            "engine": "faster-whisper",
            "model_size": "large-v3",
            "device": device,
            "vad_method": "simple",
            "enable_speaker_diarization": False
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Create engine with GPU settings
            engine = TranscriptionEngine(
                model_size=config['model_size'],
                device=config['device'],
                engine=config['engine'],
                vad_method=config['vad_method'],
                enable_speaker_diarization=config['enable_speaker_diarization'],
                compute_type="float16" if config['device'] == "cuda" else "float32"
            )
            
            # Clear GPU memory before transcription
            if gpu_available:
                torch.cuda.empty_cache()
            
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
            
            # Save the best result
            if len(text) > 5000:  # If we get a substantial transcription
                with open(f'transcript_output/gpu_test_{i}.txt', 'w', encoding='utf-8') as f:
                    f.write(f"Test: {config['name']}\n")
                    f.write(f"Processing time: {processing_time:.2f}s\n")
                    f.write(f"Text length: {len(text)} characters\n")
                    f.write(f"Coverage: {(last_segment.get('end', 0) / duration) * 100:.1f}%\n")
                    f.write("-" * 50 + "\n")
                    f.write(text)
                print(f"💾 Saved to transcript_output/gpu_test_{i}.txt")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def update_configuration_for_gpu():
    """Update the configuration for GPU acceleration."""
    
    print("🔧 Updating configuration for GPU acceleration...")
    
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
    
    # Whisper configuration - GPU optimized
    whisper_model: str = "large-v3"  # Use large-v3 for better accuracy
    language: str = "en"
    device: str = "cuda"  # Use GPU for better performance
    
    # Transcription engine options - Use WhisperX for GPU
    transcription_engine: str = "whisperx"  # Best for GPU acceleration
    
    # VAD (Voice Activity Detection) options
    vad_method: str = "simple"  # Options: "simple", "webrtcvad", "silero"
    enable_vad: bool = False  # Disabled for long audio files
    
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
    
    # Performance - GPU optimized
    max_concurrent_jobs: int = 2  # Reduced for GPU memory management
    job_timeout: int = 3600  # 1 hour for long files
    
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
    
    print("✅ Configuration updated for GPU!")
    print("   - Changed device to cuda")
    print("   - Changed engine to whisperx (best for GPU)")
    print("   - Using large-v3 model")
    print("   - Disabled VAD for long audio files")
    print("   - Increased max file size to 500MB")
    print("   - Reduced concurrent jobs to 2 for GPU memory")

def create_gpu_optimized_script():
    """Create a GPU-optimized transcription script."""
    
    script_content = '''#!/usr/bin/env python3
"""
GPU-optimized transcription script for long audio files.
"""

import os
import sys
import torch
import librosa
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine

def transcribe_with_gpu(audio_path: str, output_path: str = None):
    """Transcribe audio file using GPU acceleration."""
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not available, falling back to CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Create GPU-optimized engine
    engine = TranscriptionEngine(
        model_size="large-v3",
        device=device,
        engine="whisperx" if device == "cuda" else "whisper",
        vad_method="simple",
        enable_speaker_diarization=False,
        compute_type="float16" if device == "cuda" else "float32"
    )
    
    print(f"🚀 Starting transcription with {engine.engine} on {device}...")
    
    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
    
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
        coverage = (last_segment.get('end', 0) / duration) * 100
        print(f"📊 Coverage: {coverage:.1f}%")
        
        if coverage < 90:
            print("⚠️  WARNING: Low coverage - transcription may be incomplete!")
    
    # Save result
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcription Results\\n")
            f.write(f"Audio: {audio_path}\\n")
            f.write(f"Duration: {duration:.2f}s\\n")
            f.write(f"Processing time: {processing_time:.2f}s\\n")
            f.write(f"Coverage: {coverage:.1f}%\\n")
            f.write("-" * 50 + "\\n")
            f.write(text)
        print(f"💾 Saved to: {output_path}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gpu_transcribe.py <audio_file> [output_file]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"transcript_output/gpu_transcript_{Path(audio_file).stem}.txt"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    result = transcribe_with_gpu(audio_file, output_file)
    print("🎉 Transcription completed!")
'''
    
    with open('gpu_transcribe.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created gpu_transcribe.py script!")

if __name__ == "__main__":
    print("🚀 GPU Transcription Issue Fixer")
    print("=" * 50)
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Test transcription with GPU
    print("\n🧪 Testing transcription with GPU acceleration...")
    test_transcription_with_gpu()
    
    # Update configuration
    print("\n" + "="*60)
    update_configuration_for_gpu()
    
    # Create GPU script
    print("\n" + "="*60)
    create_gpu_optimized_script()
    
    print("\n" + "="*60)
    print("📋 Next steps:")
    print("1. Restart the transcription service")
    print("2. Use the new gpu_transcribe.py script for GPU transcription")
    print("3. The GPU should provide much faster processing")
    print("4. Monitor GPU memory usage during transcription") 