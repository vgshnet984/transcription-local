#!/usr/bin/env python3
"""
Test script for optimized transcription performance with CUDA.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine
from config import settings

def test_cuda_availability():
    """Test CUDA availability and memory."""
    print("🔍 Testing CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        return True
    else:
        print("❌ CUDA is not available")
        return False

def test_engine_initialization():
    """Test engine initialization with optimized settings."""
    print("\n🚀 Testing engine initialization...")
    
    try:
        # Test with CUDA optimization
        engine = TranscriptionEngine(
            model_size="large-v3",
            device="cuda",
            engine="faster-whisper",
            suppress_logs=False
        )
        
        print(f"✅ Engine initialized successfully")
        print(f"   Model: {engine.model_size}")
        print(f"   Device: {engine.device}")
        print(f"   Engine: {engine.engine}")
        print(f"   Compute type: {engine.compute_type}")
        
        return engine
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return None

def test_model_info(engine):
    """Test model information."""
    print("\n📊 Model Information:")
    
    info = engine.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

def test_transcription_performance(engine, audio_file):
    """Test transcription performance."""
    print(f"\n🎵 Testing transcription performance with {audio_file}...")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    start_time = time.time()
    
    try:
        result = engine.transcribe(audio_file, language="en")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Transcription completed")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Text length: {len(result.get('text', ''))} characters")
        print(f"   Engine used: {result.get('actual_engine_used', 'unknown')}")
        print(f"   Device used: {result.get('actual_device_used', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        
        # Show first 200 characters of transcription
        text = result.get('text', '')
        if text:
            print(f"   Preview: {text[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 Optimized Transcription Performance Test")
    print("=" * 50)
    
    # Test CUDA availability
    cuda_available = test_cuda_availability()
    
    if not cuda_available:
        print("\n⚠️  CUDA not available, but continuing with CPU...")
    
    # Test engine initialization
    engine = test_engine_initialization()
    
    if not engine:
        print("❌ Cannot continue without engine")
        return
    
    # Test model info
    test_model_info(engine)
    
    # Test transcription with sample audio
    sample_audio = "examples/sample_audio/sample.wav"  # Adjust path as needed
    
    if os.path.exists(sample_audio):
        test_transcription_performance(engine, sample_audio)
    else:
        print(f"\n⚠️  Sample audio not found: {sample_audio}")
        print("   Please provide a test audio file path")
        
        # Look for any audio files in the project
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        for root, dirs, files in os.walk('.'):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_path = os.path.join(root, file)
                    print(f"   Found: {audio_path}")
                    test_transcription_performance(engine, audio_path)
                    break
            else:
                continue
            break

if __name__ == "__main__":
    main() 