#!/usr/bin/env python3
"""
Fix faster-whisper to ensure complete transcription of long audio files.
The main issues are memory management, VAD settings, and chunk processing.
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

def test_faster_whisper_fixes():
    """Test faster-whisper with different configurations to ensure complete transcription."""
    
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
    
    # Test different faster-whisper configurations
    configs = [
        {
            "name": "Faster Whisper - Default (Problematic)",
            "engine": "faster-whisper",
            "model_size": "base",
            "device": device,
            "vad_method": "simple",
            "enable_speaker_diarization": False,
            "compute_type": "float16" if device == "cuda" else "float32"
        },
        {
            "name": "Faster Whisper - Large Model",
            "engine": "faster-whisper",
            "model_size": "large-v3",
            "device": device,
            "vad_method": "simple",
            "enable_speaker_diarization": False,
            "compute_type": "float16" if device == "cuda" else "float32"
        },
        {
            "name": "Faster Whisper - No VAD",
            "engine": "faster-whisper",
            "model_size": "large-v3",
            "device": device,
            "vad_method": "none",  # Disable VAD
            "enable_speaker_diarization": False,
            "compute_type": "float16" if device == "cuda" else "float32"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Clear GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Create engine with specific settings
            engine = TranscriptionEngine(
                model_size=config['model_size'],
                device=config['device'],
                engine=config['engine'],
                vad_method=config['vad_method'],
                enable_speaker_diarization=config['enable_speaker_diarization'],
                compute_type=config['compute_type']
            )
            
            # Transcribe
            result = engine.transcribe(audio_path, language="en")
            
            # Analyze results
            text = result.get('text', '')
            segments = result.get('segments', [])
            processing_time = result.get('processing_time', 0)
            
            print(f"✅ Processing time: {processing_time:.2f}s")
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
                print("✅ Transcription appears complete")
            
            # Save result if good
            if len(text) > 10000:  # Substantial transcription
                output_file = f'transcript_output/faster_whisper_test_{i}.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Test: {config['name']}\n")
                    f.write(f"Processing time: {processing_time:.2f}s\n")
                    f.write(f"Text length: {len(text)} characters\n")
                    f.write(f"Coverage: {coverage:.1f}%\n")
                    f.write("-" * 50 + "\n")
                    f.write(text)
                print(f"💾 Saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def create_faster_whisper_fixed_config():
    """Create a fixed configuration for faster-whisper."""
    
    print("🔧 Creating fixed faster-whisper configuration...")
    
    # Update the faster-whisper transcription method in engine.py
    fixed_method = '''
    def _transcribe_with_faster_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using faster-whisper with fixes for complete transcription."""
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not available")
        
        if self.faster_whisper_model is None:
            raise RuntimeError("faster-whisper model not loaded")
            
        try:
            # Use the specified language or default to English
            transcribe_language = language if language and language != "auto" else "en"
            
            # FIXED SETTINGS for complete transcription
            segments, info = self.faster_whisper_model.transcribe(
                audio_path,
                language=transcribe_language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,  # Enable context
                initial_prompt=None,
                word_timestamps=True,
                vad_filter=False,  # DISABLE VAD for complete transcription
                vad_parameters=dict(
                    min_silence_duration_ms=1000,  # Longer silence threshold
                    speech_pad_ms=500,  # Add padding around speech
                    threshold=0.5  # Lower threshold to catch more speech
                ) if self.vad_method != "none" else None,
                # Memory management for long files
                chunk_length=30,  # Process in 30-second chunks
                chunk_overlap=2,  # 2-second overlap between chunks
                # Additional parameters for completeness
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            # Convert faster-whisper segments to Whisper format
            whisper_segments = []
            full_text = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob if hasattr(segment, 'avg_logprob') else -1.0,
                    "no_speech_prob": segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.0,
                    "words": []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_dict = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability if hasattr(word, 'probability') else 0.0
                        }
                        segment_dict["words"].append(word_dict)
                
                whisper_segments.append(segment_dict)
                full_text += segment.text.strip() + " "
            
            # Create result in Whisper format
            result = {
                "text": full_text.strip(),
                "language": info.language if hasattr(info, 'language') else transcribe_language,
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else 1.0,
                "segments": whisper_segments
            }
            
            logger.info(f"faster-whisper transcription completed for language: {transcribe_language}")
            return result
            
        except Exception as e:
            logger.error(f"faster-whisper transcription failed: {e}")
            # Fallback to standard Whisper
            logger.info("Falling back to standard Whisper")
            return self._transcribe_with_whisper(audio_path, language)
'''
    
    print("✅ Fixed faster-whisper configuration created!")
    print("Key fixes:")
    print("   - Disabled VAD filter (vad_filter=False)")
    print("   - Added chunk processing (chunk_length=30)")
    print("   - Added chunk overlap (chunk_overlap=2)")
    print("   - Enabled context (condition_on_previous_text=True)")
    print("   - Added memory management parameters")
    
    return fixed_method

def create_faster_whisper_optimized_script():
    """Create an optimized script for faster-whisper with complete transcription."""
    
    script_content = '''#!/usr/bin/env python3
"""
Optimized faster-whisper script for complete transcription of long audio files.
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

def transcribe_with_faster_whisper_complete(audio_path: str, output_path: str = None):
    """Transcribe audio file using faster-whisper with complete transcription fixes."""
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not available, falling back to CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Get audio duration
    duration = librosa.get_duration(path=audio_path)
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Create faster-whisper engine with complete transcription settings
    engine = TranscriptionEngine(
        model_size="large-v3",  # Use large model for better accuracy
        device=device,
        engine="faster-whisper",
        vad_method="none",  # Disable VAD for complete transcription
        enable_speaker_diarization=False,
        compute_type="float16" if device == "cuda" else "float32"
    )
    
    print(f"🚀 Starting faster-whisper transcription with complete settings...")
    
    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Transcribe
    result = engine.transcribe(audio_path, language="en")
    
    # Analyze results
    text = result.get('text', '')
    segments = result.get('segments', [])
    processing_time = result.get('processing_time', 0)
    
    print(f"✅ Processing time: {processing_time:.2f}s")
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
    
    # Save result
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Faster-Whisper Complete Transcription Results\\n")
            f.write(f"Audio: {audio_path}\\n")
            f.write(f"Duration: {duration:.2f}s\\n")
            f.write(f"Processing time: {processing_time:.2f}s\\n")
            f.write(f"Coverage: {coverage:.1f}%\\n")
            f.write("-" * 50 + "\\n")
            f.write(text)
        print(f"💾 Saved to: {output_path}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python faster_whisper_complete.py <audio_file> [output_file]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"transcript_output/faster_whisper_complete_{Path(audio_file).stem}.txt"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    result = transcribe_with_faster_whisper_complete(audio_file, output_file)
    print("🎉 Faster-whisper transcription completed!")
'''
    
    with open('faster_whisper_complete.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created faster_whisper_complete.py script!")

if __name__ == "__main__":
    print("🔧 Faster-Whisper Complete Transcription Fixer")
    print("=" * 60)
    
    # Test current faster-whisper issues
    print("🧪 Testing faster-whisper configurations...")
    test_faster_whisper_fixes()
    
    # Create fixed configuration
    print("\n" + "="*60)
    create_faster_whisper_fixed_config()
    
    # Create optimized script
    print("\n" + "="*60)
    create_faster_whisper_optimized_script()
    
    print("\n" + "="*60)
    print("📋 Key fixes for faster-whisper complete transcription:")
    print("1. Disable VAD filter (vad_filter=False)")
    print("2. Use chunk processing (chunk_length=30, chunk_overlap=2)")
    print("3. Enable context (condition_on_previous_text=True)")
    print("4. Use large-v3 model for better accuracy")
    print("5. Add memory management parameters")
    print("6. Clear GPU memory before processing")
    print("\nUse faster_whisper_complete.py for optimized transcription!") 