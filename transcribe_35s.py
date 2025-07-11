import os
import whisperx
import torch

# Set environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

print("="*60)
print("TRANSCRIBING FIRST 35 SECONDS OF TES1.MP3")
print("="*60)

try:
    print("Loading WhisperX model...")
    model = whisperx.load_model('large-v3', device='cuda')
    print("Model loaded successfully!")
    
    print("Loading audio file...")
    audio = whisperx.load_audio('examples/sample_audio/tes1_first_35s.wav')
    
    print("Transcribing...")
    result = model.transcribe(audio)
    
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULTS")
    print("="*60)
    print(f"Detected language: {result.get('language', 'Unknown')}")
    print(f"Number of segments: {len(result['segments'])}")
    print(f"Total duration: {result['segments'][-1]['end'] if result['segments'] else 0:.2f} seconds")
    
    print("\n" + "-"*60)
    print("TRANSCRIPT TEXT:")
    print("-"*60)
    
    # Print full transcript by concatenating segments
    full_text = " ".join([segment['text'] for segment in result['segments']])
    print(full_text)
    
    print("\n" + "-"*60)
    print("SEGMENTED TRANSCRIPT:")
    print("-"*60)
    
    # Print segments with timestamps
    for i, segment in enumerate(result['segments']):
        print(f"Segment {i+1} ({segment['start']:.2f}s - {segment['end']:.2f}s):")
        print(f"  Text: {segment['text']}")
        print(f"  Confidence: {segment.get('avg_logprob', 'N/A'):.3f}")
        print()
    
    print("="*60)
    print("TRANSCRIPTION COMPLETED!")
    print("="*60)
    
except Exception as e:
    print(f"Error during transcription: {e}")
    import traceback
    traceback.print_exc() 