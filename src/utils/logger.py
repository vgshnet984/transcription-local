#!/usr/bin/env python3
"""
Comprehensive logging utility for transcription platform.
Captures timing, configuration, performance metrics, and saves transcripts.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
import uuid
import sys

class TranscriptionLogger:
    """Comprehensive logger for transcription runs with detailed metrics."""
    
    def __init__(self, log_dir: str = "logs", transcript_dir: str = "transcript_output"):
        self.log_dir = Path(log_dir)
        self.transcript_dir = Path(transcript_dir)
        self.session_id = str(uuid.uuid4())
        self.start_time = None
        self.run_data = {}
        
        # Ensure directories exist
        self.log_dir.mkdir(exist_ok=True)
        self.transcript_dir.mkdir(exist_ok=True)
        
        # Setup detailed logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Remove default handler
        logger.remove()
        
        # Add console handler with minimal output
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # Add detailed log file
        detailed_log_path = self.log_dir / f"detailed_{datetime.now().strftime('%Y%m%d')}.log"
        logger.add(
            detailed_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days"
        )
        
        # Add session-specific log
        session_log_path = self.log_dir / f"session_{self.session_id}.log"
        logger.add(
            session_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            level="INFO"
        )
    
    def start_run(self, config: Dict[str, Any], audio_file: str) -> str:
        """Start a new transcription run and return run ID."""
        self.start_time = time.time()
        run_id = str(uuid.uuid4())
        
        self.run_data = {
            "run_id": run_id,
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "audio_file": audio_file,
            "config": config,
            "system_info": self._get_system_info(),
            "performance_metrics": {}
        }
        
        logger.info(f"🚀 Starting transcription run {run_id}")
        logger.info(f"📁 Audio file: {audio_file}")
        logger.info(f"⚙️  Configuration: {json.dumps(config, indent=2)}")
        
        return run_id
    
    def log_vad_results(self, vad_method: str, segments: List[Dict], processing_time: float):
        """Log VAD processing results."""
        self.run_data["vad_results"] = {
            "method": vad_method,
            "segments_count": len(segments),
            "total_speech_duration": sum(seg.get("duration", 0) for seg in segments),
            "processing_time": processing_time,
            "segments": segments
        }
        
        logger.info(f"🎤 VAD ({vad_method}): {len(segments)} segments, "
                   f"{self.run_data['vad_results']['total_speech_duration']:.2f}s speech, "
                   f"{processing_time:.2f}s processing")
    
    def log_transcription_start(self, engine: str, model: str):
        """Log transcription start."""
        self.run_data["transcription"] = {
            "engine": engine,
            "model": model,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info(f"🎯 Starting transcription with {engine} ({model})")
    
    def log_transcription_complete(self, result: Dict[str, Any], processing_time: float):
        """Log transcription completion and save transcript."""
        if "transcription" not in self.run_data:
            self.run_data["transcription"] = {}
        
        self.run_data["transcription"].update({
            "end_time": datetime.now().isoformat(),
            "processing_time": processing_time,
            "result": result
        })
        
        # Calculate metrics
        text = result.get("text", "")
        segments = result.get("segments", [])
        confidence = result.get("confidence", 0.0)
        
        metrics = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "segments_count": len(segments),
            "confidence": confidence,
            "processing_time": processing_time,
            "words_per_second": len(text.split()) / processing_time if processing_time > 0 else 0
        }
        
        self.run_data["performance_metrics"] = metrics
        
        logger.info(f"✅ Transcription completed: {len(text.split())} words, "
                   f"{confidence:.2f} confidence, {processing_time:.2f}s")
        
        # Save transcript
        self._save_transcript(result)
    
    def log_error(self, error: Exception, stage: str):
        """Log errors with context."""
        error_data = {
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        if "errors" not in self.run_data:
            self.run_data["errors"] = []
        self.run_data["errors"].append(error_data)
        
        logger.error(f"❌ Error in {stage}: {error}")
    
    def end_run(self, success: bool = True):
        """End the transcription run and save comprehensive log."""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.run_data["total_time"] = total_time
            self.run_data["success"] = success
            self.run_data["end_time"] = datetime.now().isoformat()
        
        # Save run log
        self._save_run_log()
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status} Run completed in {total_time:.2f}s")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for logging."""
        import torch
        import psutil
        
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": os.name,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        
        # GPU info if available
        if torch.cuda.is_available():
            system_info["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        else:
            system_info["gpu"] = {"available": False}
        
        return system_info
    
    def _save_transcript(self, result: Dict[str, Any]):
        """Save transcript to file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = self.run_data.get("run_id", "unknown")
        
        # Create filename
        audio_name = Path(self.run_data["audio_file"]).stem
        filename = f"{audio_name}_{run_id}_{timestamp}.txt"
        filepath = self.transcript_dir / filename
        
        # Prepare transcript content
        content = []
        content.append("=" * 80)
        content.append("TRANSCRIPTION RESULT")
        content.append("=" * 80)
        content.append(f"Run ID: {run_id}")
        content.append(f"Timestamp: {datetime.now().isoformat()}")
        content.append(f"Audio File: {self.run_data['audio_file']}")
        content.append(f"Engine: {self.run_data.get('transcription', {}).get('engine', 'unknown')}")
        content.append(f"Model: {self.run_data.get('transcription', {}).get('model', 'unknown')}")
        content.append(f"Confidence: {result.get('confidence', 0.0):.3f}")
        content.append(f"Processing Time: {self.run_data.get('transcription', {}).get('processing_time', 0.0):.2f}s")
        content.append("")
        
        # Main transcript text
        content.append("TRANSCRIPT:")
        content.append("-" * 40)
        content.append(result.get("text", ""))
        content.append("")
        
        # Segments with timestamps
        segments = result.get("segments", [])
        if segments:
            content.append("SEGMENTS:")
            content.append("-" * 40)
            for i, segment in enumerate(segments, 1):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                content.append(f"[{i:2d}] {start:6.2f}s - {end:6.2f}s: {text}")
        
        content.append("")
        content.append("=" * 80)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"💾 Transcript saved: {filepath}")
    
    def _save_run_log(self):
        """Save comprehensive run log to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = self.run_data.get("run_id", "unknown")
        filename = f"run_{run_id}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Run log saved: {filepath}")

# Global logger instance
transcription_logger = TranscriptionLogger()

def get_transcription_logger() -> TranscriptionLogger:
    """Get the global transcription logger instance."""
    return transcription_logger 