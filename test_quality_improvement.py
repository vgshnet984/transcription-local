#!/usr/bin/env python3
"""
Test script to demonstrate transcription quality improvements
Compares different settings and shows quality differences
"""

import os
import time
from src.transcription.engine import TranscriptionEngine, check_cudnn_installation

def test_quality_settings():
    """Test different quality settings and compare results"""
    
    print("=" * 80)
    print("TRANSCRIPTION QUALITY COMPARISON TEST")
    print("=" * 80)
    
    # Check CUDA/cuDNN setup
    cudnn_installed, cudnn_path = check_cudnn_installation()
    if cudnn_installed:
        print(f"✅ CUDA/cuDNN available: {cudnn_path}")
        device = "cuda"
    else:
        print("⚠️  CUDA/cuDNN not found, using CPU")
        device = "cpu"
    
    # Test audio file (use the same audio that generated tespp7101.txt)
    audio_file = "uploads/tespp7101.wav"  # Adjust path as needed
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide the correct path to your test audio file")
        return
    
    print(f"🎵 Testing with audio: {audio_file}")
    print()
    
    # Test configurations
    configs = [
        {
            "name": "CURRENT SETTINGS (Poor Quality)",
            "settings": {
                "model_size": "base",
                "device": device,
                "engine": "faster-whisper",
                "vad_method": "none",
                "enable_speaker_diarization": True,
                "suppress_logs": True
            }
        },
        {
            "name": "RECOMMENDED SETTINGS (High Quality)",
            "settings": {
                "model_size": "large-v3",
                "device": device,
                "engine": "whisperx",
                "vad_method": "silero",
                "enable_speaker_diarization": True,
                "suppress_logs": False
            }
        },
        {
            "name": "BALANCED SETTINGS (Good Quality)",
            "settings": {
                "model_size": "large",
                "device": device,
                "engine": "whisperx",
                "vad_method": "webrtcvad",
                "enable_speaker_diarization": True,
                "suppress_logs": True
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"🧪 TEST {i}: {config['name']}")
        print("-" * 60)
        
        try:
            # Create engine with settings
            engine = TranscriptionEngine(**config['settings'])
            
            # Transcribe
            start_time = time.time()
            result = engine.transcribe(audio_file)
            end_time = time.time()
            
            # Analyze results
            text = result.get("text", "")
            confidence = result.get("confidence", 0)
            speakers = result.get("speakers", [])
            
            # Quality metrics
            word_count = len(text.split())
            unique_speakers = len(set([s.get("id", "Unknown") for s in speakers]))
            avg_sentence_length = word_count / max(1, text.count('.') + text.count('!') + text.count('?'))
            
            # Store results
            test_result = {
                "name": config['name'],
                "text": text,
                "confidence": confidence,
                "speakers": speakers,
                "word_count": word_count,
                "unique_speakers": unique_speakers,
                "avg_sentence_length": avg_sentence_length,
                "processing_time": end_time - start_time,
                "settings": config['settings']
            }
            results.append(test_result)
            
            # Display summary
            print(f"✅ Processing time: {test_result['processing_time']:.2f}s")
            print(f"📊 Confidence: {confidence:.2f}")
            print(f"👥 Speakers detected: {unique_speakers}")
            print(f"📝 Words: {word_count}")
            print(f"📏 Avg sentence length: {avg_sentence_length:.1f} words")
            print()
            
            # Show sample text
            sample_text = text[:300] + "..." if len(text) > 300 else text
            print("📄 Sample text:")
            print(f"   {sample_text}")
            print()
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            print()
    
    # Compare results
    print("=" * 80)
    print("QUALITY COMPARISON SUMMARY")
    print("=" * 80)
    
    for result in results:
        print(f"\n🔍 {result['name']}")
        print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
        print(f"   📊 Confidence: {result['confidence']:.2f}")
        print(f"   👥 Speakers: {result['unique_speakers']}")
        print(f"   📝 Words: {result['word_count']}")
        print(f"   📏 Avg sentence: {result['avg_sentence_length']:.1f} words")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if results:
        best_result = max(results, key=lambda x: x['confidence'])
        print(f"🏆 Best quality: {best_result['name']}")
        print(f"   Confidence: {best_result['confidence']:.2f}")
        print(f"   Speakers: {best_result['unique_speakers']}")
        
        print("\n📋 Optimal settings for your use case:")
        optimal_settings = best_result['settings']
        for key, value in optimal_settings.items():
            print(f"   {key}: {value}")
    
    print("\n💡 For maximum quality, use:")
    print("   - Engine: WhisperX")
    print("   - Model: Large-v3")
    print("   - VAD: Silero")
    print("   - Speaker Diarization: Enabled")
    print("   - Audio Filtering: Enabled")

def test_specific_audio():
    """Test with a specific audio file"""
    print("🎵 Enter the path to your audio file:")
    audio_path = input("Path: ").strip().strip('"')
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    print(f"\n🧪 Testing with optimal settings...")
    
    # Use optimal settings
    engine = TranscriptionEngine(
        model_size="large-v3",
        device="cuda" if check_cudnn_installation()[0] else "cpu",
        engine="whisperx",
        vad_method="silero",
        enable_speaker_diarization=True,
        suppress_logs=False
    )
    
    start_time = time.time()
    result = engine.transcribe(audio_path)
    end_time = time.time()
    
    print(f"\n✅ Transcription completed in {end_time - start_time:.2f}s")
    print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
    print(f"👥 Speakers: {len(result.get('speakers', []))}")
    
    print("\n📄 Full transcript:")
    print("=" * 60)
    print(result.get('text', ''))
    print("=" * 60)

def main():
    """Main function"""
    print("🎯 Transcription Quality Improvement Test")
    print("Choose an option:")
    print("1. Run quality comparison test")
    print("2. Test specific audio file with optimal settings")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_quality_settings()
    elif choice == "2":
        test_specific_audio()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 