#!/usr/bin/env python3
"""
Test script for NVIDIA Parakeet integration.
This script tests the Parakeet model loading and transcription capabilities.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.transcription.engine import TranscriptionEngine
from src.config import settings
import torch

def test_parakeet_integration():
    """Test NVIDIA Parakeet integration."""
    print("🧪 Testing NVIDIA Parakeet Integration")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test audio file
    test_audio = "examples/sample_audio/tes1.mp3"
    if not os.path.exists(test_audio):
        print(f"❌ Test audio file not found: {test_audio}")
        print("Please ensure you have a test audio file available.")
        return False
    
    try:
        # Initialize Parakeet engine
        print("\n🚀 Initializing NVIDIA Parakeet engine...")
        start_time = time.time()
        
        engine = TranscriptionEngine(
            engine="parakeet",
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_speaker_diarization=False  # Parakeet is English-only, no need for diarization
        )
        
        load_time = time.time() - start_time
        print(f"✅ Parakeet engine loaded in {load_time:.2f}s")
        
        # Get model info
        model_info = engine.get_model_info()
        print(f"📊 Model Info:")
        print(f"   Engine: {model_info['current_engine']}")
        print(f"   Device: {model_info['current_device']}")
        print(f"   Parakeet Available: {model_info['parakeet_available']}")
        
        # Test transcription
        print(f"\n🎤 Testing transcription with {test_audio}...")
        transcription_start = time.time()
        
        result = engine.transcribe(test_audio, language="en")
        
        transcription_time = time.time() - transcription_start
        total_time = time.time() - start_time
        
        # Display results
        print(f"\n📝 Transcription Results:")
        print(f"   Text: {result['text'][:200]}...")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Processing Time: {transcription_time:.2f}s")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Actual Engine Used: {result['actual_engine_used']}")
        print(f"   Actual Model Used: {result['actual_model_used']}")
        
        # Check if Parakeet was actually used
        if result['actual_engine_used'] == 'parakeet':
            print("✅ SUCCESS: Parakeet engine was used for transcription!")
            return True
        else:
            print(f"⚠️  WARNING: Expected Parakeet but got {result['actual_engine_used']}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parakeet_performance():
    """Test Parakeet performance compared to other engines."""
    print("\n🏃 Performance Comparison Test")
    print("=" * 50)
    
    test_audio = "examples/sample_audio/tes1.mp3"
    if not os.path.exists(test_audio):
        print(f"❌ Test audio file not found: {test_audio}")
        return
    
    engines = ["parakeet", "faster-whisper", "whisper"]
    results = {}
    
    for engine_name in engines:
        try:
            print(f"\n🔄 Testing {engine_name}...")
            start_time = time.time()
            
            engine = TranscriptionEngine(
                engine=engine_name,
                model_size="base" if engine_name != "parakeet" else "base",  # Parakeet doesn't use model_size
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_speaker_diarization=False
            )
            
            result = engine.transcribe(test_audio, language="en")
            processing_time = time.time() - start_time
            
            results[engine_name] = {
                'time': processing_time,
                'text_length': len(result['text']),
                'actual_engine': result['actual_engine_used']
            }
            
            print(f"   ✅ {engine_name}: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ {engine_name}: Failed - {e}")
            results[engine_name] = None
    
    # Display comparison
    print(f"\n📊 Performance Comparison:")
    print(f"{'Engine':<15} {'Time (s)':<10} {'Text Length':<12} {'Actual Engine':<15}")
    print("-" * 60)
    
    for engine_name, result in results.items():
        if result:
            print(f"{engine_name:<15} {result['time']:<10.2f} {result['text_length']:<12} {result['actual_engine']:<15}")
        else:
            print(f"{engine_name:<15} {'FAILED':<10} {'N/A':<12} {'N/A':<15}")

def main():
    """Main test function."""
    print("NVIDIA Parakeet Integration Test")
    print("=" * 60)
    
    # Test basic integration
    success = test_parakeet_integration()
    
    if success:
        # Test performance comparison
        test_parakeet_performance()
        
        print(f"\n🎉 All tests completed!")
        print(f"💡 Parakeet is now available as an engine option:")
        print(f"   - CLI: python scripts/transcribe_cli_ultra_fast.py --engine parakeet")
        print(f"   - Web UI: Select 'NVIDIA Parakeet (Fast English)' from dropdown")
        print(f"   - API: Set engine parameter to 'parakeet'")
    else:
        print(f"\n❌ Integration test failed. Check the error messages above.")

if __name__ == "__main__":
    main() 