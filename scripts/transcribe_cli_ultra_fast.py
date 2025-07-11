#!/usr/bin/env python3
"""
Ultra-Fast CLI Transcription Tool with faster-whisper and batch processing
Optimized for maximum speed and memory efficiency

Usage:
    python scripts/transcribe_cli_ultra_fast.py --input audio.wav --language en --model large-v3
    python scripts/transcribe_cli_ultra_fast.py --input folder/ --batch --batch-size 8
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable colorama to prevent import issues
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"

from src.transcription.engine import TranscriptionEngine
from src.config import settings

def setup_ultra_fast_engine(config: Dict[str, Any]) -> TranscriptionEngine:
    """Setup ultra-fast transcription engine with faster-whisper"""
    print(f"🚀 Initializing ULTRA-FAST {config['transcription_engine']} engine...")
    print(f"Model: {config['whisper_model']}")
    print(f"Device: {config['device']}")
    print(f"Language: {config['language']}")
    print(f"Compute Type: {config['compute_type']}")
    print(f"CPU Threads: {config['cpu_threads']}")
    print(f"Batch Size: {config.get('batch_size', 'N/A')}")
    print("=" * 60)
    
    return TranscriptionEngine(
        engine=config['transcription_engine'],
        model_size=config['whisper_model'],
        device=config['device'],
        vad_method=config['vad_method'],
        enable_speaker_diarization=config.get('enable_speaker_diarization', False),
        show_romanized_text=config.get('show_romanized_text', False),
        compute_type=config['compute_type'],
        cpu_threads=config['cpu_threads']
    )

def get_audio_files(input_path: str) -> List[str]:
    """Get list of audio files from input path"""
    path = Path(input_path)
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    
    if path.is_file():
        if path.suffix.lower() in audio_extensions:
            return [str(path)]
        else:
            raise ValueError(f"File {path} is not a supported audio format")
    
    elif path.is_dir():
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(path.glob(f"*{ext}"))
            audio_files.extend(path.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            raise ValueError(f"No audio files found in directory {path}")
        
        return [str(f) for f in sorted(audio_files)]
    
    else:
        raise ValueError(f"Path {path} does not exist")

def process_single_file(file_path: str, engine: TranscriptionEngine, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process single file with timing"""
    start_time = time.time()
    
    print(f"🎵 Processing: {Path(file_path).name}")
    print("-" * 50)
    
    try:
        # Process audio
        result = engine.transcribe(
            file_path,
            language=config['language']
        )
        
        processing_time = time.time() - start_time
        
        # Add metadata
        result.update({
            'processing_time': processing_time,
            'file_path': file_path,
            'engine_used': config['transcription_engine'],
            'model_used': config['whisper_model'],
            'device_used': config['device']
        })
        
        # Print results
        print(f"✅ Completed in {processing_time:.2f}s")
        print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
        print(f"📝 Text length: {len(result.get('text', ''))} characters")
        print(f"⚡ Speed: {result.get('processing_time', 0):.2f}x real-time")
        
        return {
            'success': True,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'processing_time': processing_time,
            'engine_used': result.get('engine_used', 'unknown'),
            'model_used': result.get('model_used', 'unknown'),
            'device_used': result.get('device_used', 'unknown')
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error processing {file_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def process_batch_files(audio_files: List[str], engine: TranscriptionEngine, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process multiple files with batch processing"""
    batch_size = config.get('batch_size', 4)
    max_workers = config.get('max_workers', 2)
    
    print(f"🔄 Starting batch processing of {len(audio_files)} files")
    print(f"📦 Batch size: {batch_size}, Max workers: {max_workers}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Use engine's batch processing method
    results = engine.transcribe_batch(
        audio_files, 
        language=config['language'],
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    total_time = time.time() - start_time
    
    # Print batch summary
    successful = sum(1 for r in results if r.get('success', True) and not r.get('error'))
    failed = len(results) - successful
    
    print(f"\n📊 BATCH PROCESSING COMPLETE")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Average time per file: {total_time/len(audio_files):.2f}s")
    
    return results

def process_streaming_file(file_path: str, engine: TranscriptionEngine, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process single file with streaming transcription (chunked processing)"""
    import librosa
    import tempfile
    import soundfile as sf
    
    start_time = time.time()
    chunk_size = config.get('chunk_size', 10)
    
    print(f"🌊 Processing with streaming: {Path(file_path).name}")
    print(f"📦 Chunk size: {chunk_size}s")
    print("-" * 50)
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        duration = len(audio) / sr
        
        print(f"📊 Audio duration: {duration:.1f}s")
        print(f"🔢 Total chunks: {int(duration // chunk_size) + 1}")
        print()
        
        # Process in chunks
        all_segments = []
        full_text = ""
        chunk_count = 0
        
        for start_time_chunk in range(0, int(duration), chunk_size):
            chunk_count += 1
            end_time_chunk = min(start_time_chunk + chunk_size, int(duration))
            
            # Extract chunk
            start_sample = int(start_time_chunk * sr)
            end_sample = int(end_time_chunk * sr)
            audio_chunk = audio[start_sample:end_sample]
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_chunk, sr)
                temp_path = temp_file.name
            
            try:
                # Transcribe chunk
                chunk_start = time.time()
                result = engine.transcribe(temp_path, language=config['language'])
                chunk_time = time.time() - chunk_start
                
                # Extract text and segments
                chunk_text = result.get('text', '').strip()
                chunk_segments = result.get('segments', [])
                
                # Adjust timestamps for chunk position
                for segment in chunk_segments:
                    segment['start'] += start_time_chunk
                    segment['end'] += start_time_chunk
                
                all_segments.extend(chunk_segments)
                full_text += chunk_text + " "
                
                # Print progress
                print(f"📦 Chunk {chunk_count}: {start_time_chunk}s-{end_time_chunk}s")
                print(f"   ⏱️  Processing: {chunk_time:.2f}s")
                print(f"   📝 Text: {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
                print()
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
        
        total_time = time.time() - start_time
        
        # Create final result
        final_result = {
            'text': full_text.strip(),
            'segments': all_segments,
            'language': config['language'],
            'confidence': 0.8,  # Approximate for streaming
            'processing_time': total_time,
            'file_path': file_path,
            'engine_used': config['transcription_engine'],
            'model_used': config['whisper_model'],
            'device_used': config['device'],
            'streaming': True,
            'chunk_size': chunk_size,
            'total_chunks': chunk_count
        }
        
        print(f"✅ Streaming completed in {total_time:.2f}s")
        print(f"📝 Final text length: {len(full_text.strip())} characters")
        print(f"⚡ Speed: {total_time:.2f}x real-time")
        
        return {
            'success': True,
            'text': full_text.strip(),
            'confidence': 0.8,
            'processing_time': total_time,
            'engine_used': config['transcription_engine'],
            'model_used': config['whisper_model'],
            'device_used': config['device'],
            'streaming': True
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error processing {file_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def process_streaming_file_with_speakers(file_path: str, engine: TranscriptionEngine, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process single file with streaming transcription AND speaker preservation (two-pass approach)"""
    import librosa
    import tempfile
    import soundfile as sf
    
    start_time = time.time()
    chunk_size = config.get('chunk_size', 10)
    
    print(f"🌊 Processing with streaming + speaker preservation: {Path(file_path).name}")
    print(f"📦 Chunk size: {chunk_size}s")
    print(f"🎤 Two-pass approach: diarization first, then chunked transcription")
    print("-" * 60)
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        duration = len(audio) / sr
        
        print(f"📊 Audio duration: {duration:.1f}s")
        print(f"🔢 Total chunks: {int(duration // chunk_size) + 1}")
        print()
        
        # PASS 1: Full audio diarization
        print("🎤 PASS 1: Running speaker diarization on full audio...")
        diarization_start = time.time()
        
        # Create engine with diarization enabled
        diarization_engine = TranscriptionEngine(
            engine=config['transcription_engine'],
            model_size=config['whisper_model'],
            device=config['device'],
            vad_method=config['vad_method'],
            enable_speaker_diarization=True,  # Force diarization
            show_romanized_text=config.get('show_romanized_text', False),
            compute_type=config['compute_type'],
            cpu_threads=config['cpu_threads']
        )
        
        # Run diarization on full audio
        diarization_result = diarization_engine.transcribe(file_path, language=config['language'])
        speaker_segments = diarization_result.get('speaker_segments', [])
        diarization_time = time.time() - diarization_start
        
        print(f"✅ Diarization completed in {diarization_time:.2f}s")
        print(f"👥 Found {len(speaker_segments)} speaker segments")
        
        # Create speaker timeline for mapping
        speaker_timeline = []
        for segment in speaker_segments:
            speaker_timeline.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': segment['speaker']
            })
        
        # PASS 2: Chunked transcription with speaker mapping
        print("\n📝 PASS 2: Chunked transcription with speaker mapping...")
        
        all_segments = []
        full_text = ""
        chunk_count = 0
        
        for start_time_chunk in range(0, int(duration), chunk_size):
            chunk_count += 1
            end_time_chunk = min(start_time_chunk + chunk_size, int(duration))
            
            # Extract chunk
            start_sample = int(start_time_chunk * sr)
            end_sample = int(end_time_chunk * sr)
            audio_chunk = audio[start_sample:end_sample]
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_chunk, sr)
                temp_path = temp_file.name
            
            try:
                # Transcribe chunk
                chunk_start = time.time()
                result = engine.transcribe(temp_path, language=config['language'])
                chunk_time = time.time() - chunk_start
                
                # Extract text and segments
                chunk_text = result.get('text', '').strip()
                chunk_segments = result.get('segments', [])
                
                # Map speakers to chunk segments
                mapped_segments = []
                for segment in chunk_segments:
                    # Adjust timestamps for chunk position
                    segment_start = segment['start'] + start_time_chunk
                    segment_end = segment['end'] + start_time_chunk
                    
                    # Find which speaker was active during this segment
                    active_speaker = None
                    max_overlap = 0
                    
                    for speaker_seg in speaker_timeline:
                        # Calculate overlap
                        overlap_start = max(segment_start, speaker_seg['start'])
                        overlap_end = min(segment_end, speaker_seg['end'])
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            active_speaker = speaker_seg['speaker']
                    
                    # Create mapped segment
                    mapped_segment = {
                        'start': segment_start,
                        'end': segment_end,
                        'text': segment.get('text', ''),
                        'speaker': active_speaker or 'Unknown',
                        'confidence': segment.get('avg_logprob', 0.0)
                    }
                    mapped_segments.append(mapped_segment)
                
                all_segments.extend(mapped_segments)
                full_text += chunk_text + " "
                
                # Print progress with speaker info
                print(f"📦 Chunk {chunk_count}: {start_time_chunk}s-{end_time_chunk}s")
                print(f"   ⏱️  Processing: {chunk_time:.2f}s")
                print(f"   📝 Text: {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
                
                # Show speaker info for this chunk
                chunk_speakers = set(seg['speaker'] for seg in mapped_segments if seg['speaker'] != 'Unknown')
                if chunk_speakers:
                    print(f"   👥 Speakers: {', '.join(chunk_speakers)}")
                print()
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
        
        total_time = time.time() - start_time
        
        # Create final result with speaker labels
        speaker_labeled_text = ""
        for segment in all_segments:
            if segment['text'].strip():
                speaker_labeled_text += f"[{segment['speaker']}] {segment['text'].strip()} "
        
        print(f"✅ Streaming with speaker preservation completed in {total_time:.2f}s")
        print(f"📝 Final text length: {len(full_text.strip())} characters")
        print(f"👥 Speaker-labeled text length: {len(speaker_labeled_text.strip())} characters")
        print(f"⚡ Speed: {total_time:.2f}x real-time")
        
        return {
            'success': True,
            'text': speaker_labeled_text.strip(),
            'original_text': full_text.strip(),
            'confidence': 0.8,
            'processing_time': total_time,
            'engine_used': config['transcription_engine'],
            'model_used': config['whisper_model'],
            'device_used': config['device'],
            'streaming': True,
            'speaker_preserved': True,
            'speaker_segments': speaker_segments,
            'total_chunks': chunk_count
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error processing {file_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def save_results(results: List[Dict[str, Any]], output_file: str, config: Dict[str, Any]):
    """Save results to file"""
    output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output_data = {
        'config': config,
        'summary': {
            'total_files': len(results),
            'successful': sum(1 for r in results if r.get('success', True) and not r.get('error')),
            'failed': sum(1 for r in results if r.get('error')),
            'total_processing_time': sum(r.get('processing_time', 0) for r in results),
            'average_processing_time': sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
        },
        'results': results
    }
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Ultra-Fast CLI Transcription Tool with faster-whisper')
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--language', default='en', help='Language code (e.g., en, ta, sa)')
    parser.add_argument('--engine', choices=['faster-whisper', 'whisper', 'whisperx', 'parakeet', 'parakeet-nemo', 'wav2vec2'], default='faster-whisper', help='Transcription engine')
    parser.add_argument('--model', default='large-v3', help='Model size (base, small, medium, large, large-v3)')
    parser.add_argument('--device', default='cuda', help='Device (cpu, cuda)')
    parser.add_argument('--compute-type', choices=['float16', 'float32', 'int8'], default='float16', help='Compute type for faster-whisper')
    parser.add_argument('--cpu-threads', type=int, default=8, help='Number of CPU threads')
    parser.add_argument('--vad', default='simple', help='VAD method (simple, silero)')
    parser.add_argument('--speaker-diarization', action='store_true', help='Enable speaker diarization')
    parser.add_argument('--romanized', action='store_true', help='Show romanized text for Indian languages')
    parser.add_argument('--output', help='Output file path (JSON format)')
    
    # Batch processing options
    parser.add_argument('--batch', action='store_true', help='Enable batch processing for multiple files')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=2, help='Maximum parallel workers')
    
    # Streaming options
    parser.add_argument('--stream', action='store_true', help='Enable streaming transcription (chunked processing)')
    parser.add_argument('--chunk-size', type=int, default=10, help='Audio chunk size in seconds for streaming')
    parser.add_argument('--preserve-speakers', action='store_true', help='Preserve speaker continuity across chunks (two-pass approach)')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'transcription_engine': args.engine,
        'whisper_model': args.model,
        'device': args.device,
        'language': args.language,
        'compute_type': args.compute_type,
        'cpu_threads': args.cpu_threads,
        'vad_method': args.vad,
        'enable_speaker_diarization': args.speaker_diarization,
        'show_romanized_text': args.romanized,
        'batch_size': args.batch_size,
        'max_workers': args.max_workers,
        'stream': args.stream,
        'chunk_size': args.chunk_size,
        'preserve_speakers': args.preserve_speakers,
        'output_file': args.output
    }
    
    try:
        # Get audio files
        audio_files = get_audio_files(args.input)
        print(f"📁 Found {len(audio_files)} audio file(s)")
        
        # Setup engine
        engine = setup_ultra_fast_engine(config)
        
        # Process files
        if args.batch and len(audio_files) > 1:
            results = process_batch_files(audio_files, engine, config)
        elif args.stream:
            results = []
            for file_path in audio_files:
                if args.preserve_speakers:
                    result = process_streaming_file_with_speakers(file_path, engine, config)
                else:
                    result = process_streaming_file(file_path, engine, config)
                results.append(result)
                print()  # Add spacing between files
        else:
            results = []
            for file_path in audio_files:
                result = process_single_file(file_path, engine, config)
                results.append(result)
                print()  # Add spacing between files
        
        # Save results if output file specified
        if args.output:
            save_results(results, args.output, config)
        
        # Print final summary
        successful = sum(1 for r in results if r.get('success', True) and not r.get('error'))
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        print(f"\n🎉 TRANSCRIPTION COMPLETE!")
        print(f"✅ Successfully processed: {successful}/{len(results)} files")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        if successful > 0:
            print(f"⚡ Average speed: {total_time/len(audio_files):.2f}s per file")
        
        return 0 if successful == len(results) else 1
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 