#!/usr/bin/env python3
"""
Test script for speaker diarization with Hugging Face models.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_diarization_setup():
    """Test if diarization is properly configured."""
    print("Testing Speaker Diarization Setup")
    print("=" * 40)
    
    # Check HF token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("✗ HF_TOKEN not set")
        print("Run: python scripts/setup_huggingface.py")
        return False
    
    print("✓ HF_TOKEN is set")
    
    # Test pyannote import
    try:
        from pyannote.audio import Pipeline
        print("✓ pyannote.audio imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pyannote.audio: {e}")
        return False
    
    # Test model loading
    try:
        print("Loading speaker diarization model...")
        start_time = time.time()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        load_time = time.time() - start_time
        print(f"✓ Speaker diarization model loaded in {load_time:.2f}s")
        
        print("Loading voice activity detection model...")
        start_time = time.time()
        vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection-3.1",
            use_auth_token=token
        )
        load_time = time.time() - start_time
        print(f"✓ Voice activity detection model loaded in {load_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you've accepted the model licenses on Hugging Face")
        print("2. Check your internet connection")
        print("3. Verify your HF_TOKEN is correct")
        return False

def test_with_sample_audio():
    """Test diarization with a sample audio file."""
    print("\nTesting with sample audio...")
    
    # Look for sample audio files
    sample_dir = Path("examples/sample_audio")
    if not sample_dir.exists():
        print("No sample audio directory found")
        return False
    
    audio_files = list(sample_dir.glob("*.wav")) + list(sample_dir.glob("*.mp3"))
    if not audio_files:
        print("No sample audio files found")
        return False
    
    # Use the first audio file
    audio_file = audio_files[0]
    print(f"Using sample file: {audio_file}")
    
    try:
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        
        token = os.getenv("HF_TOKEN")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        
        print("Running diarization...")
        start_time = time.time()
        
        with ProgressHook() as hook:
            diarization = pipeline(str(audio_file), hook=hook)
        
        processing_time = time.time() - start_time
        print(f"✓ Diarization completed in {processing_time:.2f}s")
        
        # Extract speaker segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })
        
        print(f"Found {len(segments)} speaker segments:")
        for i, segment in enumerate(segments[:5]):  # Show first 5
            print(f"  {i+1}. {segment['speaker']}: {segment['start']:.1f}s - {segment['end']:.1f}s ({segment['duration']:.1f}s)")
        
        if len(segments) > 5:
            print(f"  ... and {len(segments) - 5} more segments")
        
        return True
        
    except Exception as e:
        print(f"✗ Diarization test failed: {e}")
        return False

def main():
    """Main test function."""
    if not test_diarization_setup():
        return False
    
    if not test_with_sample_audio():
        print("Note: Sample audio test failed, but setup is working")
        print("You can still use diarization with your own audio files")
    
    print("\n✓ Speaker diarization setup is working!")
    print("\nTo enable diarization in your transcription platform:")
    print("1. Set enable_speaker_diarization=True in your config")
    print("2. Make sure HF_TOKEN is set in your environment")
    print("3. The models will be downloaded automatically on first use")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 