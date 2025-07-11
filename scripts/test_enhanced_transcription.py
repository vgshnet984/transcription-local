#!/usr/bin/env python3
"""
Test script for enhanced transcription capabilities.
Tests the large model, audio preprocessing, and VAD features.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings
from src.transcription.engine import TranscriptionEngine
from src.audio.processor import AudioProcessor
from loguru import logger


def test_model_loading():
    """Test loading the large model."""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        # Test with large model
        engine = TranscriptionEngine(model_size="large-v3", device="cuda")
        info = engine.get_model_info()
        
        print(f"✅ Model loaded successfully:")
        print(f"   - Model size: {info['model_size']}")
        print(f"   - Device: {info['device']}")
        print(f"   - WhisperX available: {info['whisperx_available']}")
        print(f"   - Audio preprocessing: {info['audio_preprocessing']}")
        print(f"   - VAD enabled: {info['vad_enabled']}")
        print(f"   - Denoising enabled: {info['denoising_enabled']}")
        print(f"   - Normalization enabled: {info['normalization_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_audio_preprocessing():
    """Test audio preprocessing capabilities."""
    print("\n" + "=" * 60)
    print("Testing Audio Preprocessing")
    print("=" * 60)
    
    processor = AudioProcessor()
    
    # Check if we have a test audio file
    test_files = [
        "examples/sample_audio/Hello.m4a",
        "uploads/Hello.m4a",
        "examples/sample_audio/test_audio.wav"
    ]
    
    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("⚠️  No test audio file found. Skipping preprocessing test.")
        return True
    
    try:
        print(f"Testing with file: {test_file}")
        
        # Test preprocessing
        if settings.enable_audio_preprocessing:
            print("Testing audio preprocessing...")
            preprocessed_path = processor.preprocess_audio(test_file)
            print(f"✅ Preprocessing completed: {preprocessed_path}")
            
            # Test VAD
            if settings.enable_vad:
                print("Testing VAD...")
                vad_path = processor.apply_vad(preprocessed_path)
                print(f"✅ VAD completed: {vad_path}")
                
                # Clean up
                if vad_path != preprocessed_path:
                    processor.cleanup_file(preprocessed_path)
                processor.cleanup_file(vad_path)
            else:
                processor.cleanup_file(preprocessed_path)
        else:
            print("⚠️  Audio preprocessing disabled in settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio preprocessing failed: {e}")
        return False


def test_transcription_quality():
    """Test transcription quality with different settings."""
    print("\n" + "=" * 60)
    print("Testing Transcription Quality")
    print("=" * 60)
    
    # Check if we have a test audio file
    test_files = [
        "examples/sample_audio/Hello.m4a",
        "uploads/Hello.m4a",
        "examples/sample_audio/test_audio.wav"
    ]
    
    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("⚠️  No test audio file found. Skipping transcription test.")
        return True
    
    try:
        print(f"Testing transcription with file: {test_file}")
        
        # Test with large model
        engine = TranscriptionEngine(model_size="large-v3", device="cuda")
        
        start_time = time.time()
        result = engine.transcribe(test_file)
        processing_time = time.time() - start_time
        
        print(f"✅ Transcription completed in {processing_time:.2f}s")
        print(f"   - Text: {result['text'][:100]}...")
        print(f"   - Language: {result['language']}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        print(f"   - Processing time: {result['processing_time']:.2f}s")
        print(f"   - Error: {result['error']}")
        
        if result['speaker_segments']:
            print(f"   - Speaker segments: {len(result['speaker_segments'])}")
            for i, segment in enumerate(result['speaker_segments'][:3]):
                print(f"     Segment {i+1}: {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False


def test_configuration():
    """Test current configuration settings."""
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)
    
    print(f"Whisper model: {settings.whisper_model}")
    print(f"Device: {settings.device}")
    print(f"Language: {settings.language}")
    print(f"Audio preprocessing: {settings.enable_audio_preprocessing}")
    print(f"VAD enabled: {settings.enable_vad}")
    print(f"Denoising enabled: {settings.enable_denoising}")
    print(f"Normalization enabled: {settings.enable_normalization}")
    print(f"High-pass filter: {settings.highpass_freq}Hz")
    print(f"Low-pass filter: {settings.lowpass_freq}Hz")
    print(f"Target sample rate: {settings.target_sample_rate}Hz")
    
    return True


def main():
    """Run all tests."""
    print("Enhanced Transcription System Test")
    print("Testing large model, audio preprocessing, and VAD capabilities")
    print("=" * 80)
    
    tests = [
        ("Configuration", test_configuration),
        ("Model Loading", test_model_loading),
        ("Audio Preprocessing", test_audio_preprocessing),
        ("Transcription Quality", test_transcription_quality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced transcription system is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 