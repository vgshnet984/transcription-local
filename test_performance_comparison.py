#!/usr/bin/env python3
"""
Performance comparison script for different transcription engines and settings.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine

def test_engine_performance(engine_name, model_size, device, audio_file):
    """Test performance of a specific engine configuration."""
    print(f"\n🔧 Testing {engine_name} with {model_size} on {device}...")
    
    try:
        # Initialize engine
        start_init = time.time()
        engine = TranscriptionEngine(
            model_size=model_size,
            device=device,
            engine=engine_name,
            suppress_logs=True
        )
        init_time = time.time() - start_init
        
        # Test transcription
        start_transcribe = time.time()
        result = engine.transcribe(audio_file, language="en")
        transcribe_time = time.time() - start_transcribe
        
        # Calculate metrics
        text_length = len(result.get('text', ''))
        confidence = result.get('confidence', 0)
        chars_per_second = text_length / transcribe_time if transcribe_time > 0 else 0
        
        print(f"   ✅ Success")
        print(f"   Init time: {init_time:.2f}s")
        print(f"   Transcribe time: {transcribe_time:.2f}s")
        print(f"   Text length: {text_length} chars")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Speed: {chars_per_second:.1f} chars/sec")
        
        return {
            'engine': engine_name,
            'model': model_size,
            'device': device,
            'init_time': init_time,
            'transcribe_time': transcribe_time,
            'text_length': text_length,
            'confidence': confidence,
            'chars_per_second': chars_per_second,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            'engine': engine_name,
            'model': model_size,
            'device': device,
            'success': False,
            'error': str(e)
        }

def main():
    """Main comparison function."""
    print("🏁 Transcription Performance Comparison")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Find test audio file
    audio_file = None
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_file = os.path.join(root, file)
                break
        if audio_file:
            break
    
    if not audio_file:
        print("❌ No audio file found for testing")
        return
    
    print(f"Using audio file: {audio_file}")
    
    # Test configurations
    configurations = [
        # CPU configurations
        ("whisper", "base", "cpu"),
        ("whisper", "small", "cpu"),
        ("whisper", "medium", "cpu"),
        
        # CUDA configurations (if available)
        ("whisper", "base", "cuda") if cuda_available else None,
        ("whisper", "small", "cuda") if cuda_available else None,
        ("whisper", "medium", "cuda") if cuda_available else None,
        ("whisper", "large-v3", "cuda") if cuda_available else None,
        
        # Faster-whisper configurations
        ("faster-whisper", "base", "cpu"),
        ("faster-whisper", "small", "cpu"),
        ("faster-whisper", "medium", "cpu"),
        ("faster-whisper", "base", "cuda") if cuda_available else None,
        ("faster-whisper", "small", "cuda") if cuda_available else None,
        ("faster-whisper", "medium", "cuda") if cuda_available else None,
        ("faster-whisper", "large-v3", "cuda") if cuda_available else None,
        
        # WhisperX configurations
        ("whisperx", "base", "cpu"),
        ("whisperx", "small", "cpu"),
        ("whisperx", "medium", "cpu"),
        ("whisperx", "base", "cuda") if cuda_available else None,
        ("whisperx", "small", "cuda") if cuda_available else None,
        ("whisperx", "medium", "cuda") if cuda_available else None,
        ("whisperx", "large-v3", "cuda") if cuda_available else None,
    ]
    
    # Filter out None configurations
    configurations = [config for config in configurations if config is not None]
    
    results = []
    
    # Run tests
    for engine_name, model_size, device in configurations:
        result = test_engine_performance(engine_name, model_size, device, audio_file)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful tests")
        return
    
    # Sort by speed (chars per second)
    successful_results.sort(key=lambda x: x['chars_per_second'], reverse=True)
    
    print(f"{'Engine':<15} {'Model':<10} {'Device':<6} {'Time(s)':<8} {'Speed':<12} {'Confidence':<10}")
    print("-" * 70)
    
    for result in successful_results:
        print(f"{result['engine']:<15} {result['model']:<10} {result['device']:<6} "
              f"{result['transcribe_time']:<8.2f} {result['chars_per_second']:<12.1f} "
              f"{result['confidence']:<10.3f}")
    
    # Find best performers
    fastest = successful_results[0]
    most_accurate = max(successful_results, key=lambda x: x['confidence'])
    
    print(f"\n🏆 Fastest: {fastest['engine']} {fastest['model']} on {fastest['device']} "
          f"({fastest['chars_per_second']:.1f} chars/sec)")
    print(f"🎯 Most Accurate: {most_accurate['engine']} {most_accurate['model']} on {most_accurate['device']} "
          f"(confidence: {most_accurate['confidence']:.3f})")
    
    # CUDA vs CPU comparison
    cuda_results = [r for r in successful_results if r['device'] == 'cuda']
    cpu_results = [r for r in successful_results if r['device'] == 'cpu']
    
    if cuda_results and cpu_results:
        avg_cuda_speed = sum(r['chars_per_second'] for r in cuda_results) / len(cuda_results)
        avg_cpu_speed = sum(r['chars_per_second'] for r in cpu_results) / len(cpu_results)
        speedup = avg_cuda_speed / avg_cpu_speed if avg_cpu_speed > 0 else 0
        
        print(f"\n⚡ CUDA vs CPU Performance:")
        print(f"   Average CUDA speed: {avg_cuda_speed:.1f} chars/sec")
        print(f"   Average CPU speed: {avg_cpu_speed:.1f} chars/sec")
        print(f"   CUDA speedup: {speedup:.1f}x")

if __name__ == "__main__":
    main() 