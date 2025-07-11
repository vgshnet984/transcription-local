#!/usr/bin/env python3
"""
Debug script to test transcription and identify encoding issues
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.transcription.engine import TranscriptionEngine

def debug_transcription():
    """Debug transcription to identify encoding issues."""
    sample_audio = "examples/sample_audio/tes1_first_35s.wav"
    
    print("Debugging transcription...")
    print(f"Sample audio: {sample_audio}")
    
    if not os.path.exists(sample_audio):
        print(f"Sample audio not found: {sample_audio}")
        return
    
    # Create engine
    engine = TranscriptionEngine(
        engine="whisper",
        model_size="base",
        device="cpu"  # Use CPU to avoid CUDA issues
    )
    
    # Transcribe
    print("Starting transcription...")
    try:
        result = engine.transcribe(sample_audio, language="en")
        
        print("Transcription completed!")
        print(f"Text length: {len(result.get('text', ''))}")
        
        # Check for problematic characters
        text = result.get('text', '')
        print("First 100 characters:")
        print(repr(text[:100]))
        
        # Try to print with different encodings
        print("\nTrying different print methods:")
        
        # Method 1: Direct print
        try:
            print("Method 1 (direct):", text[:50])
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Encode/decode
        try:
            encoded = text.encode('utf-8', errors='ignore')
            decoded = encoded.decode('utf-8')
            print("Method 2 (encode/decode):", decoded[:50])
        except Exception as e:
            print(f"Method 2 failed: {e}")
        
        # Method 3: Replace problematic chars
        try:
            safe_text = text.encode('ascii', errors='ignore').decode('ascii')
            print("Method 3 (ASCII only):", safe_text[:50])
        except Exception as e:
            print(f"Method 3 failed: {e}")
        
        # Save to file
        output_file = "transcripts/debug_output.txt"
        os.makedirs("transcripts", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(f"Debug Transcription Output\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Text: {text}\n")
        
        print(f"\nSaved to: {output_file}")
        
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transcription() 