#!/usr/bin/env python3
"""
Fast CLI Transcription Tool - No Database, File Output Only
Optimized for Indian accents with large models

Usage:
    python scripts/transcribe_cli_fast.py --input audio.wav --language ta --model large-v3
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable colorama to prevent import issues
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"

from src.transcription.engine import TranscriptionEngine
from src.config import settings

def setup_engine(config: Dict[str, Any]) -> TranscriptionEngine:
    """Setup transcription engine with minimal overhead"""
    print(f"Initializing {config['transcription_engine']} engine...")
    print(f"Model: {config['whisper_model']}")
    print(f"Device: {config['device']}")
    print(f"Language: {config['language']}")
    print(f"VAD: {config['vad_method']}")
    print(f"Speaker Diarization: {config.get('enable_speaker_diarization', False)}")
    print("=" * 50)
    
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

def save_to_file(output_path: str, result: Dict[str, Any], config: Dict[str, Any]):
    """Save transcription result to file"""
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(f"Transcription Results\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"File: {result.get('file_path', 'Unknown')}\n")
            f.write(f"Engine: {result.get('engine_used', 'Unknown')}\n")
            f.write(f"Model: {result.get('model_used', 'Unknown')}\n")
            f.write(f"Device: {result.get('device_used', 'Unknown')}\n")
            f.write(f"Language: {config.get('language', 'Unknown')}\n")
            f.write(f"Confidence: {result.get('confidence', 0):.2%}\n")
            f.write(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
            f.write(f"Speaker Diarization: {config.get('enable_speaker_diarization', False)}\n")
            f.write(f"=" * 50 + "\n\n")
            
            # Write transcription text
            f.write(result.get('text', ''))
            
        print(f"Transcript saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving to file: {e}")

def process_file(file_path: str, engine: TranscriptionEngine, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process single file with timing"""
    start_time = time.time()
    
    print(f"Processing: {Path(file_path).name}")
    print("=" * 60)
    
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
        print(f"\nTranscription completed!")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Engine used: {result.get('engine_used', 'Unknown')}")
        print(f"Text length: {len(result.get('text', ''))} characters")
        
        # Save to file if specified
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
        processing_time = time.time() - start_time
        print(f"Error processing {file_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def main():
    parser = argparse.ArgumentParser(description='Fast CLI Transcription Tool (No Database)')
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--language', default='en', help='Language code (e.g., en, ta, sa)')
    parser.add_argument('--engine', choices=['whisper', 'whisperx', 'faster-whisper', 'parakeet', 'parakeet-nemo', 'wav2vec2'], default='whisper', help='Transcription engine')
    parser.add_argument('--model', default='large-v3', help='Model size (base, small, medium, large, large-v3)')
    parser.add_argument('--device', default='cuda', help='Device (cpu, cuda)')
    parser.add_argument('--vad', default='simple', help='VAD method (simple, silero)')
    parser.add_argument('--speaker-diarization', action='store_true', help='Enable speaker diarization')
    parser.add_argument('--romanized', action='store_true', help='Show romanized text for Indian languages')
    parser.add_argument('--output', help='Output file path')
    
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
        'output_file': args.output
    }
    
    # Setup engine
    engine = setup_engine(config)
    
    # Process file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return 1
    
    if input_path.is_file():
        result = process_file(str(input_path), engine, config)
        if result['success']:
            print(f"\nSuccess! Processing completed in {result['processing_time']:.2f}s")
            return 0
        else:
            print(f"\nFailed: {result['error']}")
            return 1
    else:
        print(f"Error: {input_path} is not a file")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 