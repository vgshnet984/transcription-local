#!/usr/bin/env python3
"""
Test script for comprehensive logging system.
Tests transcription with different configurations and verifies logging output.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine
from utils.logger import get_transcription_logger

def test_transcription_with_logging(audio_path: str, config: dict):
    """Test transcription with specific configuration and logging."""
    print(f"\n🔧 Testing with config: {config}")
    print("-" * 60)
    
    try:
        # Create engine with configuration
        engine = TranscriptionEngine(
            model_size=config.get("model_size", "base"),
            engine=config.get("engine", "whisper"),
            vad_method=config.get("vad_method", "none"),
            enable_speaker_diarization=config.get("enable_speaker_diarization", False),
            device=config.get("device", "auto"),
            suppress_logs=False  # Enable logging for testing
        )
        
        # Transcribe with logging
        start_time = time.time()
        result = engine.transcribe(audio_path, language=config.get("language", "auto"))
        total_time = time.time() - start_time
        
        # Display results
        print(f"✅ Transcription completed in {total_time:.2f}s")
        print(f"📝 Text length: {len(result.get('text', ''))} characters")
        print(f"🎯 Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"🔊 Engine used: {result.get('engine_used', 'unknown')}")
        print(f"🎤 VAD method: {result.get('vad_method', 'none')}")
        
        # Show first 200 characters of transcript
        text = result.get('text', '')
        if text:
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"📄 Preview: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Comprehensive Logging System Test")
    print("=" * 60)
    
    # Find test audio file
    audio_paths = [
        "examples/sample_audio/tes1_first_35s.wav",
        "examples/sample_audio/Hello.m4a",
        "examples/sample_audio/tamil_test.m4a"
    ]
    
    audio_path = None
    for path in audio_paths:
        if os.path.exists(path):
            audio_path = path
            break
    
    if not audio_path:
        print("❌ No test audio file found!")
        return
    
    print(f"Using audio file: {audio_path}")
    
    # Test configurations
    test_configs = [
        {
            "name": "Fast (Simple VAD)",
            "model_size": "base",
            "engine": "whisper",
            "vad_method": "simple",
            "enable_speaker_diarization": False,
            "device": "auto"
        },
        {
            "name": "Quality (Silero VAD)",
            "model_size": "medium",
            "engine": "whisperx",
            "vad_method": "silero",
            "enable_speaker_diarization": False,
            "device": "auto"
        },
        {
            "name": "Best (Large Model + Speaker Diarization)",
            "model_size": "large",
            "engine": "whisperx",
            "vad_method": "silero",
            "enable_speaker_diarization": True,
            "device": "auto"
        }
    ]
    
    # Run tests
    successful_tests = 0
    total_tests = len(test_configs)
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}/{total_tests}: {config['name']}")
        
        if test_transcription_with_logging(audio_path, config):
            successful_tests += 1
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📊 Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Check log files
    print("\n📁 Generated Files:")
    log_dir = Path("logs")
    transcript_dir = Path("transcript_output")
    
    if log_dir.exists():
        log_files = list(log_dir.glob("*.json"))
        print(f"   📊 Run logs: {len(log_files)} files")
        for log_file in log_files[-3:]:  # Show last 3
            print(f"      - {log_file.name}")
    
    if transcript_dir.exists():
        transcript_files = list(transcript_dir.glob("*.txt"))
        print(f"   📝 Transcripts: {len(transcript_files)} files")
        for transcript_file in transcript_files[-3:]:  # Show last 3
            print(f"      - {transcript_file.name}")
    
    print("\n🎉 Test completed! Check the logs and transcript_output directories for detailed results.")

if __name__ == "__main__":
    main() 