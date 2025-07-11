#!/usr/bin/env python3
"""
Test script to compare different Whisper model sizes and performance.
"""

import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcription.engine import TranscriptionEngine
from src.config import settings
import torch

def test_model_performance(model_size: str, test_audio_path: str | None = None):
    """Test a specific model size and report performance metrics."""
    print(f"\n{'='*50}")
    print(f"Testing {model_size.upper()} model")
    print(f"{'='*50}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model
    start_time = time.time()
    try:
        engine = TranscriptionEngine(model_size=model_size, device=settings.device)
        load_time = time.time() - start_time
        print(f"Model load time: {load_time:.2f}s")
        
        # Get model info
        info = engine.get_model_info()
        print(f"Model info: {info}")
        
        # Test transcription if audio file provided
        if test_audio_path and os.path.exists(test_audio_path):
            print(f"\nTranscribing: {test_audio_path}")
            transcribe_start = time.time()
            result = engine.transcribe(test_audio_path)
            transcribe_time = time.time() - transcribe_start
            
            print(f"Transcription time: {transcribe_time:.2f}s")
            print(f"Text length: {len(result['text'])} characters")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
            # Show first 200 characters of transcription
            preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            print(f"Preview: {preview}")
            
        return True
        
    except Exception as e:
        print(f"Error testing {model_size} model: {e}")
        return False

def main():
    """Main function to test all model sizes."""
    print("Whisper Model Performance Test")
    print("="*50)
    
    # Check for test audio file
    test_audio = None
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
        if not os.path.exists(test_audio):
            print(f"Warning: Test audio file not found: {test_audio}")
            test_audio = None
    
    # Model sizes to test (in order of increasing size and accuracy)
    model_sizes = ["tiny", "base", "small", "medium", "large"]
    
    # Test each model
    results = {}
    for model_size in model_sizes:
        success = test_model_performance(model_size, test_audio)
        results[model_size] = success
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print("Model sizes tested successfully:")
    for model_size, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_size}")
    
    print(f"\nRecommendations:")
    print(f"  - tiny: Fastest, lowest accuracy (~39M parameters)")
    print(f"  - base: Good balance (~74M parameters)")
    print(f"  - small: Better accuracy (~244M parameters)")
    print(f"  - medium: High accuracy (~769M parameters)")
    print(f"  - large: Best accuracy (~1550M parameters)")
    
    print(f"\nWith your RTX 3060 (12GB VRAM), you can use:")
    print(f"  - All models up to 'large' should fit in VRAM")
    print(f"  - Recommended: 'base' or 'small' for good balance")
    print(f"  - Use 'large' for maximum accuracy if you have time")

if __name__ == "__main__":
    main() 