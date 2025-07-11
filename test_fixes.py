#!/usr/bin/env python3
"""
Quick test script to verify transcription fixes.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine

def test_fixes():
    """Test the transcription fixes."""
    print("🔧 Testing Transcription Fixes")
    print("=" * 40)
    
    # Find test audio file
    audio_file = None
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                audio_file = os.path.join(root, file)
                break
        if audio_file:
            break
    
    if not audio_file:
        print("❌ No audio file found")
        return
    
    print(f"Using: {audio_file}")
    
    # Test with optimized settings
    engine = TranscriptionEngine(
        model_size="base",  # Use smaller model for quick test
        device="cuda" if torch.cuda.is_available() else "cpu",
        engine="faster-whisper",
        enable_speaker_diarization=False,  # Disable for speed
        suppress_logs=False
    )
    
    start_time = time.time()
    result = engine.transcribe(audio_file, language="en")
    end_time = time.time()
    
    print(f"\n✅ Test completed in {end_time - start_time:.1f}s")
    print(f"Text length: {len(result.get('text', ''))} chars")
    print(f"Engine used: {result.get('actual_engine_used', 'unknown')}")
    print(f"Device used: {result.get('actual_device_used', 'unknown')}")
    
    # Show first 200 chars
    text = result.get('text', '')
    if text:
        print(f"Preview: {text[:200]}...")
    
    # Check for repetitive text
    if "Thank you" in text:
        thank_you_count = text.count("Thank you")
        print(f"⚠️  Found {thank_you_count} 'Thank you' repetitions")
    else:
        print("✅ No excessive 'Thank you' repetitions found")

if __name__ == "__main__":
    test_fixes() 