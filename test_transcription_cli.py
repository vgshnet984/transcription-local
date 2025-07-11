#!/usr/bin/env python3
"""
CLI Test script for transcription functionality
Tests the CLI tools with sample audio
"""

import subprocess
import sys
import os
import time

def test_cli_transcription():
    """Test CLI transcription with sample audio."""
    sample_audio = "examples/sample_audio/tes1_first_35s.wav"
    output_file = "transcripts/cli_test_output.txt"
    
    print("Testing CLI Transcription")
    print("=" * 50)
    
    # Check if sample audio exists
    if not os.path.exists(sample_audio):
        print(f"Sample audio not found: {sample_audio}")
        return False
    print(f"Sample audio found: {sample_audio}")
    
    # Ensure transcripts directory exists
    os.makedirs("transcripts", exist_ok=True)
    
    # Test with fast CLI
    print("\nTesting fast CLI transcription...")
    try:
        result = subprocess.run([
            sys.executable, "scripts/transcribe_cli_fast.py",
            "--input", sample_audio,
            "--language", "en",
            "--engine", "whisper",
            "--model", "base",
            "--output", output_file
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("Fast CLI transcription completed")
            print(result.stdout)
            
            # Check if output file was created
            if os.path.exists(output_file):
                print(f"Output file created: {output_file}")
                
                # Read and display first few lines
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')[:10]
                    print("   Preview:")
                    for line in lines:
                        if line.strip():
                            print(f"   {line}")
            else:
                print(f"Output file not found: {output_file}")
                return False
        else:
            print("Fast CLI transcription failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Fast CLI transcription timed out")
        return False
    except Exception as e:
        print(f"Fast CLI transcription error: {e}")
        return False
    
    print("\nCLI transcription test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_cli_transcription()
    exit(0 if success else 1) 