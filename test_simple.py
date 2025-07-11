#!/usr/bin/env python3
"""Simple test to verify the fixes."""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all imports work."""
    try:
        from config import settings
        print("✅ Config import works")
        
        from audio.processor import AudioProcessor
        print("✅ Audio processor import works")
        
        from transcription.engine import TranscriptionEngine
        print("✅ Engine import works")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_settings():
    """Test settings configuration."""
    try:
        from config import settings
        print(f"✅ Upload directory: {settings.upload_directory}")
        print(f"✅ Database URL: {settings.database_url}")
        print(f"✅ Max file size bytes: {settings.max_file_size_bytes}")
        print(f"✅ Max file size str: {settings.max_file_size_str}")
        return True
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

def test_audio_processor():
    """Test audio processor initialization."""
    try:
        from audio.processor import AudioProcessor
        processor = AudioProcessor()
        print(f"✅ Audio processor initialized")
        print(f"✅ Max file size: {processor.max_file_size}")
        print(f"✅ Supported formats: {processor.supported_formats}")
        return True
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing imports and settings...")
    
    if test_imports() and test_settings() and test_audio_processor():
        print("🎉 All tests passed! You can now run start_basic_ui.py")
    else:
        print("❌ Some tests failed. Check the errors above.") 