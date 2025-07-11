#!/usr/bin/env python3
"""
CLI Transcription Tool for Batch Processing

Usage:
    python scripts/transcribe_cli.py --input path/to/audio/file.wav
    python scripts/transcribe_cli.py --input path/to/folder --batch
    python scripts/transcribe_cli.py --input audio.wav --engine whisperx --model large-v3 --device cuda --language ta
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.transcription.engine import TranscriptionEngine
from src.audio.processor import AudioProcessor
from src.database.models import AudioFile, TranscriptionJob, Transcription
from src.database.init_db import get_db
from src.config import settings


def setup_engine(config: Dict[str, Any]) -> TranscriptionEngine:
    """Initialize transcription engine with given config"""
    print(f"Initializing {config['transcription_engine']} engine...")
    print(f"Model: {config['whisper_model']}")
    print(f"Device: {config['device']}")
    print(f"Language: {config['language']}")
    print(f"VAD: {config['vad_method']}")
    print(f"Speaker Diarization: {config.get('enable_speaker_diarization', False)}")
    
    return TranscriptionEngine(
        engine=config['transcription_engine'],
        model_size=config['whisper_model'],
        device=config['device'],
        vad_method=config['vad_method'],
        enable_speaker_diarization=config.get('enable_speaker_diarization', False),
        show_romanized_text=config.get('show_romanized_text', False),
        compute_type="float16" if config['device'] == 'cuda' else "float32",
        cpu_threads=8
    )


def process_single_file(file_path: str, engine: TranscriptionEngine, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single audio file"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Process audio
        result = engine.transcribe(
            file_path,
            language=config['language']
        )
        
        processing_time = time.time() - start_time
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            return {'success': False, 'error': result['error']}
        
        # Print results
        print(f"\n✅ Transcription completed in {processing_time:.2f}s")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Engine used: {result.get('engine_used', 'unknown')}")
        print(f"Model used: {result.get('model_used', 'unknown')}")
        print(f"Device used: {result.get('device_used', 'unknown')}")
        
        print(f"\n📝 Transcription:")
        print("-" * 40)
        print(result.get('text', ''))
        print("-" * 40)
        
        # Save to database if requested
        if not config.get('no_db', False):
            save_to_database(file_path, result, config)
        
        # Save to file if output path specified
        if config.get('output_file'):
            save_to_file(config['output_file'], result, config)
        
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
        print(f"❌ Error processing {file_path}: {str(e)}")
        return {'success': False, 'error': str(e)}


def save_to_file(output_path: str, result: Dict[str, Any], config: Dict[str, Any]):
    """Save transcription result to file"""
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output content
        content = f"""Transcription Results
{'='*50}

File Information:
- Engine: {result.get('engine_used', 'unknown')}
- Model: {result.get('model_used', 'unknown')}
- Device: {result.get('device_used', 'unknown')}
- Language: {config.get('language', 'unknown')}
- Confidence: {result.get('confidence', 0):.2%}
- Processing Time: {result.get('processing_time', 0):.2f}s

Configuration:
- Transcription Engine: {config.get('transcription_engine', 'unknown')}
- Whisper Model: {config.get('whisper_model', 'unknown')}
- VAD Method: {config.get('vad_method', 'unknown')}
- Speaker Diarization: {config.get('enable_speaker_diarization', False)}
- Show Romanized: {config.get('show_romanized_text', False)}

Transcription:
{'='*50}
{result.get('text', '')}
{'='*50}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        
        print(f"💾 Saved transcription to: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save to file: {str(e)}")


def save_to_database(file_path: str, result: Dict[str, Any], config: Dict[str, Any]):
    """Save transcription result to database"""
    if config.get('no_db', False):
        return  # Skip DB if --no-db is set
    try:
        db_gen = get_db()
        db = next(db_gen)
        try:
            # Create audio file record
            audio_file = AudioFile(
                filename=os.path.basename(file_path),
                path=file_path,
                size=os.path.getsize(file_path),
                duration=result.get('duration', 0),
                format=Path(file_path).suffix[1:],
                status='processed'
            )
            db.add(audio_file)
            db.flush()
            
            # Create transcription job
            job = TranscriptionJob(
                audio_file_id=audio_file.id,
                status='completed',
                progress=100.0,
                config=config
            )
            db.add(job)
            db.flush()
            
            # Create transcription record
            transcription = Transcription(
                job_id=job.id,
                text=result.get('text', ''),
                confidence=result.get('confidence', 0),
                processing_time=result.get('processing_time', 0)
            )
            db.add(transcription)
            db.commit()
            
            print(f"💾 Saved to database - Audio ID: {audio_file.id}, Job ID: {job.id}")
        finally:
            db.close()
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save to database: {str(e)}")


def process_batch(input_path: str, engine: TranscriptionEngine, config: Dict[str, Any]):
    """Process multiple files in batch"""
    path_obj = Path(input_path)
    
    if path_obj.is_file():
        files = [path_obj]
    elif path_obj.is_dir():
        # Get all audio files
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        files = [f for f in path_obj.iterdir() 
                if f.is_file() and f.suffix.lower() in audio_extensions]
        files.sort()
    else:
        print(f"❌ Error: {input_path} is not a valid file or directory")
        return
    
    if not files:
        print(f"❌ No audio files found in {input_path}")
        return
    
    print(f"🎵 Found {len(files)} audio file(s) to process")
    
    results = []
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"\n📁 File {i}/{len(files)}")
        result = process_single_file(str(file_path), engine, config)
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(files)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='CLI Transcription Tool')
    
    # Input
    parser.add_argument('--input', '-i', required=True,
                       help='Input audio file or directory for batch processing')
    
    # Engine configuration
    parser.add_argument('--engine', choices=['whisper', 'whisperx', 'faster-whisper'], default='whisper',
                       help='Transcription engine (default: whisper)')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3'], 
                       default='base', help='Whisper model size (default: base)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use (default: cpu)')
    parser.add_argument('--language', '-l', default='en',
                       help='Language code (default: en)')
    parser.add_argument('--vad', choices=['simple', 'webrtc', 'silero'], default='simple',
                       help='Voice Activity Detection method (default: simple)')
    
    # Features
    parser.add_argument('--speaker-diarization', action='store_true',
                       help='Enable speaker diarization')
    parser.add_argument('--romanized', action='store_true',
                       help='Show romanized text instead of native script')
    
    # Output
    parser.add_argument('--output', '-o',
                       help='Output file for results (JSON format)')
    parser.add_argument('--no-db', action='store_true',
                       help='Do not save results to database (file/console only)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'transcription_engine': args.engine,
        'whisper_model': args.model,
        'device': args.device,
        'language': args.language,
        'vad_method': args.vad,
        'enable_speaker_diarization': args.speaker_diarization,
        'show_romanized_text': args.romanized,
        'no_db': args.no_db,
        'output_file': args.output
    }
    
    print("🎤 CLI Transcription Tool")
    print("=" * 40)
    
    # Initialize engine
    try:
        engine = setup_engine(config)
    except Exception as e:
        print(f"❌ Failed to initialize engine: {str(e)}")
        return 1
    
    # Process files
    try:
        process_batch(args.input, engine, config)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 