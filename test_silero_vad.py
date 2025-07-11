#!/usr/bin/env python3
"""
Test script for Silero VAD implementation.
Tests the VAD processor with different methods to ensure Silero VAD works correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio.vad_processor import VADProcessor, SILERO_AVAILABLE, WEBRTCVAD_AVAILABLE
from loguru import logger

def test_vad_methods(audio_path: str):
    """Test all available VAD methods."""
    print("=" * 60)
    print("Testing VAD Methods")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Test each VAD method
    methods_to_test = ["simple", "webrtcvad", "silero"]
    
    for method in methods_to_test:
        print(f"\n🔍 Testing {method.upper()} VAD...")
        
        try:
            # Create VAD processor
            start_time = time.time()
            vad_processor = VADProcessor(method=method)
            load_time = time.time() - start_time
            
            print(f"   Load time: {load_time:.2f}s")
            
            # Test VAD detection
            start_time = time.time()
            segments = vad_processor.detect_voice_activity(audio_path)
            detect_time = time.time() - start_time
            
            print(f"   Detection time: {detect_time:.2f}s")
            print(f"   Segments found: {len(segments)}")
            
            if segments:
                total_duration = sum(seg["duration"] for seg in segments)
                print(f"   Total speech duration: {total_duration:.2f}s")
                print(f"   First segment: {segments[0]['start']:.2f}s - {segments[0]['end']:.2f}s")
                print(f"   Last segment: {segments[-1]['start']:.2f}s - {segments[-1]['end']:.2f}s")
            
            print(f"   ✅ {method.upper()} VAD test completed successfully")
            
        except Exception as e:
            print(f"   ❌ {method.upper()} VAD test failed: {e}")
            logger.error(f"{method.upper()} VAD test failed: {e}")

def check_dependencies():
    """Check VAD dependencies."""
    print("=" * 60)
    print("Checking VAD Dependencies")
    print("=" * 60)
    
    print(f"Silero VAD available: {'✅' if SILERO_AVAILABLE else '❌'}")
    print(f"WebRTC VAD available: {'✅' if WEBRTCVAD_AVAILABLE else '❌'}")
    
    if not SILERO_AVAILABLE:
        print("\nTo install Silero VAD dependencies:")
        print("pip install torch torchaudio")
    
    if not WEBRTCVAD_AVAILABLE:
        print("\nTo install WebRTC VAD:")
        print("pip install webrtcvad")

def main():
    """Main test function."""
    print("Silero VAD Implementation Test")
    print("=" * 60)
    
    # Check dependencies
    check_dependencies()
    
    # Get audio file from command line or find default
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"\n❌ Audio file not found: {audio_path}")
            return
    else:
        # Find test audio file
        audio_paths = [
            "examples/sample_audio/tes1.mp3",
            "examples/sample_audio/tes1_first_35s.wav",
            "examples/sample_audio/Hello.m4a",
            "uploads/tes1.mp3",
            "test_audio.mp3",
            "sample.wav"
        ]
        
        audio_path = None
        for path in audio_paths:
            if os.path.exists(path):
                audio_path = path
                break
    
    if audio_path:
        print(f"\nUsing audio file: {audio_path}")
        test_vad_methods(audio_path)
    else:
        print("\n❌ No test audio file found!")
        print("Please place an audio file in one of these locations:")
        for path in audio_paths:
            print(f"  - {path}")
        print("\nOr provide the path as a command line argument:")
        print("  python test_silero_vad.py path/to/audio/file.wav")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main() 