#!/usr/bin/env python3
"""
Test the fixed faster-whisper configuration for complete transcription.
"""

import os
import sys
import torch
import librosa
from pathlib import Path

# Add cuDNN to PATH
os.environ['PATH'] += ";C:\\cudnn\\bin"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcription.engine import TranscriptionEngine

def test_fixed_faster_whisper():
    """Test the fixed faster-whisper configuration."""
    
    audio_path = 'uploads/c942fd8d-c949-43eb-8246-5abe431073ec.m4a'
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    try:
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Create faster-whisper engine with fixed settings
        print("🚀 Creating fixed faster-whisper engine...")
        engine = TranscriptionEngine(
            model_size="large-v3",
            device=device,
            engine="faster-whisper",
            vad_method="none",  # Disable VAD
            enable_speaker_diarization=False,
            compute_type="float16" if device == "cuda" else "float32"
        )
        
        # Transcribe
        print("🎯 Starting transcription...")
        result = engine.transcribe(audio_path, language="en")
        
        # Analyze results
        text = result.get('text', '')
        segments = result.get('segments', [])
        processing_time = result.get('processing_time', 0)
        
        print(f"\n✅ Transcription completed!")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📝 Text length: {len(text)} characters")
        print(f"📝 Word count: {len(text.split())} words")
        print(f"🎯 Segments: {len(segments)}")
        
        if segments:
            last_segment = segments[-1]
            coverage = (last_segment.get('end', 0) / duration) * 100
            print(f"📊 Coverage: {coverage:.1f}%")
            print(f"⏰ Last segment ends at: {last_segment.get('end', 0):.2f}s")
            
            if coverage >= 95:
                print("✅ EXCELLENT: Complete transcription achieved!")
            elif coverage >= 80:
                print("⚠️  GOOD: Most of audio transcribed")
            else:
                print("❌ POOR: Incomplete transcription")
        
        # Check for incomplete transcription
        if text.endswith('...') or text.endswith('..'):
            print("⚠️  WARNING: Transcription ends with ellipsis!")
        elif len(text) < 5000:
            print("⚠️  WARNING: Very short transcription!")
        else:
            print("✅ Transcription appears complete!")
        
        # Save result
        output_file = 'transcript_output/faster_whisper_fixed_test.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Fixed Faster-Whisper Test Results\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Audio duration: {duration:.2f}s\n")
            f.write(f"Processing time: {processing_time:.2f}s\n")
            f.write(f"Text length: {len(text)} characters\n")
            f.write(f"Coverage: {coverage:.1f}%\n")
            f.write("-" * 50 + "\n")
            f.write(text)
        
        print(f"💾 Saved to: {output_file}")
        
        # Show preview
        print(f"\n📖 TRANSCRIPTION PREVIEW:")
        print("-" * 50)
        print(f"First 300 characters: {text[:300]}...")
        print(f"\nLast 300 characters: ...{text[-300:]}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Testing Fixed Faster-Whisper Configuration")
    print("=" * 60)
    
    result = test_fixed_faster_whisper()
    
    if result:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!") 