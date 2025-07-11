#!/usr/bin/env python3
"""
Transcribe the latest audio file using GPU with fixed CUDA path.
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

def transcribe_latest_audio():
    """Transcribe the latest audio file in uploads directory."""
    
    # Find the latest audio file
    uploads_dir = Path("uploads")
    audio_files = list(uploads_dir.glob("*.m4a")) + list(uploads_dir.glob("*.mp3")) + list(uploads_dir.glob("*.wav"))
    
    if not audio_files:
        print("❌ No audio files found in uploads directory")
        return
    
    # Get the latest file
    latest_file = max(audio_files, key=lambda x: x.stat().st_mtime)
    audio_path = str(latest_file)
    
    print(f"🎵 Found latest audio file: {latest_file.name}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not available, falling back to CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Create GPU-optimized engine
    print("🚀 Creating transcription engine...")
    engine = TranscriptionEngine(
        model_size="large-v3",
        device=device,
        engine="whisper",  # Use standard Whisper for reliability
        vad_method="simple",
        enable_speaker_diarization=False,
        compute_type="float16" if device == "cuda" else "float32"
    )
    
    print(f"🚀 Starting transcription with {engine.engine} on {device}...")
    
    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Transcribe
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
        
        if coverage < 90:
            print("⚠️  WARNING: Low coverage - transcription may be incomplete!")
        else:
            print("✅ Excellent coverage - transcription appears complete!")
    
    # Check for incomplete transcription
    if text.endswith('...') or text.endswith('..') or text.endswith('.'):
        print("⚠️  WARNING: Transcription ends with ellipsis - may be incomplete!")
    elif len(text) < 1000:
        print("⚠️  WARNING: Very short transcription!")
    else:
        print("✅ Transcription appears complete!")
    
    # Save result
    output_dir = Path("transcript_output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"complete_transcript_{latest_file.stem}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Complete Transcription Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Audio file: {latest_file.name}\n")
        f.write(f"Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
        f.write(f"Processing time: {processing_time:.2f}s\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Engine used: {engine.engine}\n")
        f.write(f"Model used: large-v3\n")
        f.write(f"Text length: {len(text)} characters\n")
        f.write(f"Word count: {len(text.split())} words\n")
        f.write(f"Segments: {len(segments)}\n")
        if segments:
            f.write(f"Coverage: {coverage:.1f}%\n")
            f.write(f"Last segment ends at: {last_segment.get('end', 0):.2f}s\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("TRANSCRIPTION TEXT:\n")
        f.write("=" * 50 + "\n")
        f.write(text)
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("SEGMENTS WITH TIMESTAMPS:\n")
        f.write("=" * 50 + "\n")
        for i, segment in enumerate(segments[:10]):  # Show first 10 segments
            f.write(f"Segment {i+1}: {segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s\n")
            f.write(f"  Text: {segment.get('text', '')}\n\n")
        if len(segments) > 10:
            f.write(f"... and {len(segments) - 10} more segments\n")
    
    print(f"💾 Complete transcription saved to: {output_file}")
    
    # Show preview
    print(f"\n📖 TRANSCRIPTION PREVIEW:")
    print("-" * 50)
    print(f"First 300 characters: {text[:300]}...")
    print(f"\nLast 300 characters: ...{text[-300:]}")
    
    return result

if __name__ == "__main__":
    print("🎯 Complete GPU Transcription of Latest Audio")
    print("=" * 60)
    
    try:
        result = transcribe_latest_audio()
        print("\n🎉 Transcription completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc() 