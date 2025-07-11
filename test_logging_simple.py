#!/usr/bin/env python3
"""
Simple test for logging system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_logging():
    """Test the logging system."""
    try:
        from utils.logger import get_transcription_logger
        
        # Get logger
        logger = get_transcription_logger()
        print("✅ Logger created successfully")
        
        # Test basic logging
        config = {
            "model_size": "base",
            "engine": "whisper",
            "vad_method": "simple",
            "device": "cuda"
        }
        
        run_id = logger.start_run(config, "test_audio.wav")
        print(f"✅ Run started with ID: {run_id}")
        
        # Test VAD logging
        logger.log_vad_results("simple", [{"start": 0, "end": 10, "duration": 10}], 1.5)
        print("✅ VAD results logged")
        
        # Test transcription logging
        logger.log_transcription_start("whisper", "base")
        print("✅ Transcription start logged")
        
        # Test completion logging
        result = {
            "text": "This is a test transcript.",
            "confidence": 0.95,
            "segments": [{"start": 0, "end": 5, "text": "This is a test"}]
        }
        logger.log_transcription_complete(result, 2.5)
        print("✅ Transcription completion logged")
        
        # End run
        logger.end_run(success=True)
        print("✅ Run ended successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Logging System")
    print("=" * 40)
    
    if test_logging():
        print("\n✅ All logging tests passed!")
        
        # Check generated files
        log_dir = Path("logs")
        transcript_dir = Path("transcript_output")
        
        if log_dir.exists():
            log_files = list(log_dir.glob("*.json"))
            print(f"📊 Generated {len(log_files)} log files")
        
        if transcript_dir.exists():
            transcript_files = list(transcript_dir.glob("*.txt"))
            print(f"📝 Generated {len(transcript_files)} transcript files")
    else:
        print("\n❌ Logging tests failed!") 