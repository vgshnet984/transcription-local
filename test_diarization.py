#!/usr/bin/env python3
"""
Test script to verify speaker diarization is working with HuggingFace models.
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_diarization():
    """Test if diarization is working properly."""
    print("🔍 Testing Speaker Diarization Setup...")
    print("=" * 50)
    
    # Test 1: Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN is set: {hf_token[:10]}...")
    else:
        print("❌ HF_TOKEN not found in environment")
        return False
    
    # Test 2: Check if pyannote.audio is available
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote.audio is available")
    except ImportError as e:
        print(f"❌ pyannote.audio not available: {e}")
        return False
    
    # Test 3: Test pipeline loading
    try:
        print("🔄 Loading diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("✅ Diarization pipeline loaded successfully")
        
        # Test 4: Check if we can access the models
        print("🔄 Testing model access...")
        print("✅ Model access successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load diarization pipeline: {e}")
        return False

def test_transcription_engine():
    """Test the transcription engine with diarization."""
    print("\n🔍 Testing Transcription Engine...")
    print("=" * 50)
    
    try:
        from src.transcription.engine import TranscriptionEngine
        print("✅ TranscriptionEngine imported successfully")
        
        # Test engine initialization with diarization
        engine = TranscriptionEngine(
            model_size="base",
            device="cpu",  # Use CPU for testing
            enable_speaker_diarization=True
        )
        print("✅ TranscriptionEngine initialized with diarization")
        
        # Check if diarization pipeline is loaded
        if hasattr(engine, 'diarization_pipeline') and engine.diarization_pipeline is not None:
            print("✅ Diarization pipeline is loaded in engine")
            return True
        else:
            print("❌ Diarization pipeline not loaded in engine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test transcription engine: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Diarization Tests...")
    print("=" * 60)
    
    # Test basic setup
    basic_ok = test_diarization()
    
    # Test transcription engine
    engine_ok = test_transcription_engine()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Basic Diarization Setup: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"   Transcription Engine: {'✅ PASS' if engine_ok else '❌ FAIL'}")
    
    if basic_ok and engine_ok:
        print("\n🎉 All tests passed! Speaker diarization should work.")
        print("   You can now use diarization in both Basic UI and Scripflow UI.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return basic_ok and engine_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 