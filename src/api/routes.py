from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from sqlalchemy.orm import Session
from loguru import logger
import json
from fastapi.responses import JSONResponse
import traceback
import asyncio
import base64
import io
import wave
import numpy as np
from threading import Thread

from database.init_db import get_db
from database.models import AudioFile, TranscriptionJob, Transcription, TranscriptionSegment
from audio.processor import AudioProcessor
from transcription.engine import TranscriptionEngine, transcription_engine
from config import settings
from utils.storage import LocalFileStorage

router = APIRouter()
storage = LocalFileStorage()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "transcription-api",
        "version": "1.0.0"
    }

@router.get("/debug/latest-transcription")
def get_latest_transcription_debug(db: Session = Depends(get_db)):
    """Debug endpoint to get the latest transcription data."""
    try:
        # Get the latest transcription
        latest_transcription = db.query(Transcription).order_by(Transcription.id.desc()).first()
        if not latest_transcription:
            return {"error": "No transcriptions found"}
        
        # Get the job
        job = db.query(TranscriptionJob).filter(TranscriptionJob.id == latest_transcription.job_id).first()
        
        # Get the audio file
        audio_file = None
        if job:
            audio_file = db.query(AudioFile).filter(AudioFile.id == job.audio_file_id).first()
        
        # Get segments
        segments = db.query(TranscriptionSegment).filter(
            TranscriptionSegment.transcription_id == latest_transcription.id
        ).all()
        
        return {
            "transcription": {
                "id": latest_transcription.id,
                "text": latest_transcription.text,
                "confidence": latest_transcription.confidence,
                "processing_time": latest_transcription.processing_time,
                "created_at": latest_transcription.created_at.isoformat() if latest_transcription.created_at else None
            },
            "job": {
                "id": job.id if job else None,
                "status": job.status if job else None,
                "progress": job.progress if job else None,
                "audio_file_id": job.audio_file_id if job else None
            },
            "audio_file": {
                "id": audio_file.id if audio_file else None,
                "filename": audio_file.filename if audio_file else None,
                "path": audio_file.path if audio_file else None,
                "size": audio_file.size if audio_file else None,
                "duration": audio_file.duration if audio_file else None
            },
            "segments": [
                {
                    "id": segment.id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                    "confidence": segment.confidence,
                    "speaker_id": segment.speaker_id
                }
                for segment in segments
            ]
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return {"error": str(e)}


@router.post("/upload")
async def upload_file_only(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload audio file only (no transcription)."""
    try:
        logger.info(f"Starting /upload endpoint with file: {file.filename}")
        
        # Save uploaded file
        logger.info("Calling upload_audio...")
        upload_result = await upload_audio(file, db)
        if not upload_result:
            logger.error("Failed to save uploaded file in /upload endpoint")
            return JSONResponse(status_code=500, content={"error": "Failed to save uploaded file"})
        
        logger.info(f"Audio file saved with ID: {upload_result['id']}")
        return upload_result
        
    except Exception as e:
        logger.error(f"Exception in /upload: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/transcribe")
async def transcribe_file(request: Request, db: Session = Depends(get_db)):
    """Start transcription for an uploaded file."""
    try:
        body = await request.json()
        file_id = body.get("file_id")
        config = body.get("config", {})
        
        if not file_id:
            return JSONResponse(status_code=400, content={"error": "file_id is required"})
        
        logger.info(f"Starting transcription for file_id: {file_id}")
        logger.info(f"Using configuration: {config}")
        logger.info(f"Language from config: {config.get('language', 'NOT_FOUND')}")
        logger.info(f"Engine from config: {config.get('transcription_engine', 'NOT_FOUND')}")
        
        try:
            transcription_id = await start_transcription(file_id, config, db)
            logger.info(f"Transcription completed with ID: {transcription_id}")
            if not isinstance(transcription_id, int):
                logger.error(f"start_transcription did not return an int: {transcription_id}")
                return JSONResponse(status_code=500, content={"error": "Failed to create transcription"})
            return {"transcription_id": transcription_id}
        except Exception as e:
            logger.error(f"Exception in /transcribe: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Exception in /transcribe: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/files")
def list_files(db: Session = Depends(get_db)):
    """List all uploaded audio files."""
    try:
        files = db.query(AudioFile).all()
        return [
            {
                "id": f.id,
                "filename": f.filename,
                "path": f.path,
                "size": f.size,
                "format": f.format,
                "status": f.status,
                "created_at": f.created_at.isoformat() if f.created_at is not None else None,
            }
            for f in files
        ]
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}")
def get_file_info(file_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific audio file."""
    try:
        f = db.query(AudioFile).filter(AudioFile.id == file_id).first()
        if not f:
            raise HTTPException(status_code=404, detail="File not found")
        
        info = storage.get_file_info(f.path)
        if not info:
            raise HTTPException(status_code=404, detail="File missing on disk")
        
        return {
            "id": f.id,
            "filename": f.filename,
            "path": f.path,
            "size": f.size,
            "format": f.format,
            "status": f.status,
            "created_at": f.created_at.isoformat() if f.created_at is not None else None,
            "disk_info": info,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db)):
    """Delete an audio file and its associated data."""
    try:
        f = db.query(AudioFile).filter(AudioFile.id == file_id).first()
        if not f:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete from storage
        storage.delete_file(f.path)
        
        # Delete from database
        db.delete(f)
        db.commit()
        
        return {"detail": "File deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}/download")
def download_file(file_id: int, db: Session = Depends(get_db)):
    """Download an audio file."""
    try:
        f = db.query(AudioFile).filter(AudioFile.id == file_id).first()
        if not f:
            raise HTTPException(status_code=404, detail="File not found")
        
        resp = storage.serve_file(f.path, download_name=f.filename)
        if not resp:
            raise HTTPException(status_code=404, detail="File missing on disk")
        
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcriptions/{transcription_id}/download")
async def download_transcription(transcription_id: int, db: Session = Depends(get_db)):
    """Download transcription as text file with speaker labels."""
    try:
        transcription = db.query(Transcription).filter(Transcription.id == transcription_id).first()
        
        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        # Get speaker segments for additional info
        speaker_segments = db.query(TranscriptionSegment).filter(
            TranscriptionSegment.transcription_id == transcription_id
        ).all()
        
        # Create text content with speaker information
        content = f"Transcription ID: {transcription.id}\n"
        content += f"Job ID: {transcription.job_id}\n"
        content += f"Confidence: {transcription.confidence:.2%}\n"
        content += f"Created: {transcription.created_at.isoformat()}\n"
        
        # Add speaker information
        if speaker_segments:
            speakers = {}
            for segment in speaker_segments:
                speaker_id = segment.speaker_id
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "total_duration": 0.0,
                        "segment_count": 0
                    }
                duration = segment.end_time - segment.start_time
                speakers[speaker_id]["total_duration"] += duration
                speakers[speaker_id]["segment_count"] += 1
            
            content += f"\nSpeakers detected: {len(speakers)}\n"
            for speaker_id, info in speakers.items():
                content += f"- {speaker_id}: {info['segment_count']} segments, {info['total_duration']:.1f}s\n"
        
        content += f"\nTranscription (with speaker labels):\n{transcription.text}\n"
        
        from fastapi.responses import Response
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=transcription_{transcription_id}.txt"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download transcription {transcription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcriptions")
def list_transcriptions(db: Session = Depends(get_db)):
    """List all transcriptions."""
    try:
        transcriptions = db.query(Transcription).all()
        result = []
        for t in transcriptions:
            text_preview = t.text[:100] + "..." if len(t.text) > 100 else t.text
            created_at = t.created_at.isoformat() if t.created_at is not None else None
            
            result.append({
                "id": t.id,
                "job_id": t.job_id,
                "text": text_preview,
                "confidence": t.confidence,
                "created_at": created_at,
                "processing_time": t.processing_time
            })
        return result
    except Exception as e:
        logger.error(f"Failed to list transcriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: int, db: Session = Depends(get_db)):
    """Get transcription details and segments (Scripflow UI format)."""
    try:
        transcription = db.query(Transcription).filter(Transcription.id == transcription_id).first()
        
        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        # Get speaker segments
        speaker_segments = db.query(TranscriptionSegment).filter(
            TranscriptionSegment.transcription_id == transcription_id
        ).all()
        
        # Extract unique speakers
        speakers = {}
        for segment in speaker_segments:
            speaker_id = segment.speaker_id
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    "speaker_id": speaker_id,
                    "total_duration": 0.0,
                    "segment_count": 0,
                    "first_seen": segment.start_time,
                    "last_seen": segment.end_time
                }
            
            duration = segment.end_time - segment.start_time
            speakers[speaker_id]["total_duration"] += duration
            speakers[speaker_id]["segment_count"] += 1
            speakers[speaker_id]["last_seen"] = segment.end_time
        
        # Set default values for engine config
        engine_type = "whisper"
        model_size = "base"
        device = "cpu"
        
        # Get engine info from stored configuration
        try:
            engine_info = transcription_engine.get_model_info()
        except Exception as e:
            logger.warning(f"Could not get engine info: {e}")
            engine_info = {
                "engine_type": "whisper",
                "current_model": "base",
                "current_device": "cpu",
                "cuda_available": False,
                "denoising_enabled": False,
                "normalization_enabled": False,
                "vad_enabled": False,
                "enhanced_settings": {}
            }
        
        # Try to get the actual engine config used for this transcription
        job_config = {}
        try:
            # Get job directly from database to avoid relationship issues
            job = db.query(TranscriptionJob).filter(TranscriptionJob.id == transcription.job_id).first()
            if job and job.config:
                try:
                    job_config = json.loads(job.config)
                except Exception as e:
                    logger.warning(f"Could not parse job config for transcription {transcription_id}: {e}")
                    job_config = {}
        except Exception as e:
            logger.warning(f"Could not get job for transcription {transcription_id}: {e}")
            job_config = {}
        
        # If job_config is missing, just use defaults (do not fail)
        if job_config:
            engine_type = job_config.get("transcription_engine", engine_type)
            model_size = job_config.get("whisper_model", model_size)
            device = job_config.get("device", device)
            # Update engine info with actual values used
            engine_info["current_model"] = model_size
            engine_info["current_device"] = device
            engine_info["engine_type"] = engine_type
        # For now, use the configured engine info as actual (will be updated when we implement proper tracking)
        actual_engine_info = {
            "actual_engine_used": engine_type,
            "actual_model_used": model_size,
            "actual_device_used": device
        }
        return {
            "id": transcription.id,
            "job_id": transcription.job_id,
            "text": transcription.text,
            "confidence": transcription.confidence,
            "created_at": transcription.created_at.isoformat(),
            "engine_info": {
                "model": f"{engine_info.get('engine_type', 'whisper')}-{engine_info['current_model']}",
                "device": engine_info['current_device'],
                "cuda_available": engine_info['cuda_available'],
                "processing_time": f"{transcription.processing_time:.2f}s" if transcription.processing_time is not None else None,
                "enhancements": [
                    "denoising" if engine_info['denoising_enabled'] else None,
                    "normalization" if engine_info['normalization_enabled'] else None,
                    "vad" if engine_info['vad_enabled'] else None
                ],
                "optimization": engine_info['enhanced_settings'],
                "actual_engine_used": actual_engine_info.get("actual_engine_used", engine_info.get('engine_type', 'whisper')),
                "actual_model_used": actual_engine_info.get("actual_model_used", engine_info['current_model']),
                "actual_device_used": actual_engine_info.get("actual_device_used", engine_info['current_device'])
            },
            "speakers": list(speakers.values()),
            "speaker_segments": [
                {
                    "start": segment.start_time,
                    "end": segment.end_time,
                    "speaker": segment.speaker_id,
                    "confidence": segment.confidence,
                    "duration": segment.end_time - segment.start_time
                }
                for segment in speaker_segments
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transcription {transcription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/basic/jobs/{job_id}")
def get_basic_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    audio_file = db.query(AudioFile).filter(AudioFile.id == job.audio_file_id).first()
    transcription = db.query(Transcription).filter(Transcription.job_id == job.id).first()
    return {
        "id": job.id,
        "audio_file_id": job.audio_file_id,
        "status": job.status,
        "progress": job.progress,
        "transcription_id": transcription.id if transcription else None,
        "audio_file": {
            "filename": audio_file.filename if audio_file else None,
            "path": audio_file.path if audio_file else None,
            "size": audio_file.size if audio_file else None,
            "duration": audio_file.duration if audio_file else None,
            "format": audio_file.format if audio_file else None,
        } if audio_file else None,
        "created_at": job.created_at.isoformat() if getattr(job, 'created_at', None) else None,
        "completed_at": job.completed_at.isoformat() if getattr(job, 'completed_at', None) else None,
        "error_message": job.error_message if hasattr(job, 'error_message') else None,
    }

@router.get("/basic/transcriptions/{transcription_id}")
def get_basic_transcription(transcription_id: int, db: Session = Depends(get_db)):
    transcription = db.query(Transcription).filter(Transcription.id == transcription_id).first()
    if transcription is None:
        raise HTTPException(status_code=404, detail="Transcription not found")
    segments = db.query(TranscriptionSegment).filter(TranscriptionSegment.transcription_id == transcription_id).all()
    return {
        "id": transcription.id,
        "job_id": transcription.job_id,
        "text": transcription.text,
        "confidence": transcription.confidence,
        "created_at": transcription.created_at.isoformat() if getattr(transcription, 'created_at', None) else None,
        "segments": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "confidence": s.confidence,
                "speaker_id": s.speaker_id
            } for s in segments
        ]
    }


@router.get("/engine/info")
async def get_engine_info():
    """Get information about the transcription engine."""
    try:
        info = transcription_engine.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get engine info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    """List all transcription jobs."""
    try:
        jobs = db.query(TranscriptionJob).all()
        return [
            {
                "id": job.id,
                "audio_file_id": job.audio_file_id,
                "status": job.status,
                "progress": job.progress,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            for job in jobs
        ]
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
    if job is None:
        return {"success": False, "error": "Job not found"}

    # Always fetch the latest transcription for this job
    transcription = db.query(Transcription).filter(Transcription.job_id == job_id).order_by(Transcription.id.desc()).first()
    transcription_id = transcription.id if transcription is not None else None

    # Build job dict
    job_dict = {
        "id": getattr(job, "id", None),
        "audio_file_id": getattr(job, "audio_file_id", None),
        "status": getattr(job, "status", None),
        "progress": getattr(job, "progress", None),
        "config": getattr(job, "config", None),
        "created_at": getattr(job, "created_at", None),
        "completed_at": getattr(job, "completed_at", None),
        "error_message": getattr(job, "error_message", None),
        "transcription_id": transcription_id,
    }
    # Optionally include audio_file info if needed by UI
    audio_file = db.query(AudioFile).filter(AudioFile.id == job.audio_file_id).first()
    if audio_file is not None:
        job_dict["audio_file"] = {
            "id": getattr(audio_file, "id", None),
            "filename": getattr(audio_file, "filename", None),
            "path": getattr(audio_file, "path", None),
            "size": getattr(audio_file, "size", None),
            "duration": getattr(audio_file, "duration", None),
            "format": getattr(audio_file, "format", None),
        }
    return {"success": True, "job": job_dict}


async def upload_audio(file: UploadFile, db: Session) -> Optional[Dict]:
    """Upload and save audio file to database."""
    try:
        # Validate file
        if not file.filename:
            raise ValueError("No file provided")
        
        # Save uploaded file
        audio_processor = AudioProcessor()
        
        # Ensure file is at the beginning
        await file.seek(0)
        
        saved_path, file_info = audio_processor.save_uploaded_file(
            file, 
            file.filename
        )
        
        # Create database record
        audio_file = AudioFile(
            filename=file.filename,
            path=saved_path,
            size=file_info["file_size"],
            duration=file_info["duration"],
            format=file_info["format"],
            status="uploaded"
        )
        
        db.add(audio_file)
        db.commit()
        db.refresh(audio_file)
        
        return {
            "id": audio_file.id,
            "filename": audio_file.filename,
            "path": audio_file.path
        }
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return None


async def start_transcription(file_id: int, config: Dict, db: Session) -> int:
    """Start transcription with custom configuration (ASYNC VERSION)."""
    import time
    from threading import Thread
    
    try:
        # Create transcription job
        job = TranscriptionJob(
            audio_file_id=file_id,
            status="processing",
            config=json.dumps(config) if config else None
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Start transcription in background thread
        def run_transcription():
            try:
                # Get a new database session for the background thread
                from database.init_db import SessionLocal
                background_db = SessionLocal()
                
                # Get audio file
                audio_file = background_db.query(AudioFile).filter(AudioFile.id == file_id).first()
                
                # Create custom engine based on configuration
                from transcription.engine import TranscriptionEngine
                
                # Extract configuration parameters
                engine_type = config.get("transcription_engine", "whisper")
                model_size = config.get("whisper_model", "base")
                device = config.get("device", "cpu")
                vad_method = config.get("vad_method", "simple")
                enable_speaker_diarization = config.get("enable_speaker_diarization", None)
                show_romanized_text = config.get("show_romanized_text", False)
                compute_type = config.get("compute_type", "float16" if device == "cuda" else "float32")
                cpu_threads = config.get("cpu_threads", 4)
                
                logger.info(f"Background transcription: engine={engine_type}, model={model_size}, device={device}")
                
                # Create new engine instance
                custom_engine = TranscriptionEngine(
                    model_size=model_size,
                    device=device,
                    engine=engine_type,
                    vad_method=vad_method,
                    enable_speaker_diarization=enable_speaker_diarization,
                    show_romanized_text=show_romanized_text,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads
                )
                
                # Perform transcription
                language = config.get("language", settings.language)
                if language == "auto":
                    language = None
                
                logger.info(f"Starting background transcription for file: {file_id}")
                start_time = time.time()
                transcription_result = custom_engine.transcribe(audio_file.path, language=language)
                processing_time = time.time() - start_time
                
                if transcription_result.get("error"):
                    # Update job status to failed
                    background_job = background_db.query(TranscriptionJob).filter(TranscriptionJob.id == job.id).first()
                    if background_job:
                        background_job.status = "failed"
                        background_job.error_message = transcription_result["error"]
                        background_db.commit()
                    return
                
                # Save transcription to database
                transcription = Transcription(
                    job_id=job.id,
                    text=transcription_result["text"],
                    confidence=transcription_result["confidence"],
                    processing_time=processing_time
                )
                
                background_db.add(transcription)
                background_db.commit()
                background_db.refresh(transcription)
                
                # Save transcript to transcripts/ directory
                try:
                    import os
                    transcripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'transcripts')
                    os.makedirs(transcripts_dir, exist_ok=True)
                    transcript_path = os.path.join(transcripts_dir, f'transcription_{job.id}.txt')
                    with open(transcript_path, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(f"Job ID: {job.id}\n")
                        f.write(f"Confidence: {transcription.confidence:.2%}\n")
                        f.write(f"Text:\n{transcription.text}\n")
                except Exception as e:
                    logger.error(f"Failed to save transcript to file: {e}")
                
                # Save speaker segments if available
                if transcription_result.get("speaker_segments"):
                    for segment in transcription_result["speaker_segments"]:
                        transcription_segment = TranscriptionSegment(
                            transcription_id=transcription.id,
                            start_time=segment["start"],
                            end_time=segment["end"],
                            text="",
                            confidence=segment.get("confidence", 0.5),
                            speaker_id=segment["speaker"]
                        )
                        background_db.add(transcription_segment)
                    background_db.commit()
                
                # Update job status to completed
                background_job = background_db.query(TranscriptionJob).filter(TranscriptionJob.id == job.id).first()
                if background_job:
                    background_job.status = "completed"
                    background_job.progress = 100.0
                    background_db.commit()
                
                logger.info(f"Background transcription completed for file: {file_id}")
                background_db.close()
                
            except Exception as e:
                logger.error(f"Background transcription failed: {e}")
                try:
                    background_job = background_db.query(TranscriptionJob).filter(TranscriptionJob.id == job.id).first()
                    if background_job:
                        background_job.status = "failed"
                        background_job.error_message = str(e)
                        background_db.commit()
                except:
                    pass
                background_db.close()
        
        # Start background thread
        thread = Thread(target=run_transcription)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Transcription job {job.id} started in background for file: {file_id}")
        return job.id  # Return job ID instead of transcription ID
        
    except Exception as e:
        logger.error(f"Failed to start transcription: {e}")
        raise 


@router.websocket("/ws/stream-transcribe")
async def websocket_stream_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming transcription with speaker options."""
    await websocket.accept()
    try:
        # Wait for initial config message
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        single_speaker = config.get('single_speaker', False)
        preserve_speakers = config.get('preserve_speakers', False)
        language = config.get('language', 'en')
        model_size = config.get('model', 'base')
        device = config.get('device', 'cpu')
        compute_type = config.get('compute_type', 'float16' if device == 'cuda' else 'float32')
        cpu_threads = config.get('cpu_threads', 4)
        chunk_duration = config.get('chunk_size', 3.0)
        sample_rate = 16000
        chunk_samples = int(sample_rate * chunk_duration)

        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "WebSocket streaming transcription ready",
            "engine": "faster-whisper",
            "model": model_size,
            "single_speaker": single_speaker,
            "preserve_speakers": preserve_speakers
        }))

        audio_buffer = []
        speaker_timeline = None
        diarization_done = False
        temp_audio_file = None
        import tempfile
        import soundfile as sf
        import numpy as np

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "audio_chunk":
                audio_data = base64.b64decode(message["audio"])
                audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                audio_buffer.extend(audio_chunk)
                # If preserve_speakers, accumulate all audio for diarization
                if preserve_speakers:
                    if temp_audio_file is None:
                        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    sf.write(temp_audio_file.name, audio_buffer, sample_rate)
                # Process chunk if enough audio and not in preserve_speakers mode
                if not preserve_speakers and len(audio_buffer) >= chunk_samples:
                    chunk_to_process = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples:]
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        sf.write(temp_file.name, chunk_to_process, sample_rate)
                        temp_path = temp_file.name
                    try:
                        engine = TranscriptionEngine(
                            engine="faster-whisper",
                            model_size=model_size,
                            device=device,
                            compute_type=compute_type,
                            cpu_threads=cpu_threads,
                            enable_speaker_diarization=not single_speaker
                        )
                        result = engine.transcribe(temp_path, language=language)
                        text = result.get("text", "").strip()
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": text,
                            "confidence": result.get("confidence", 0.0),
                            "timestamp": asyncio.get_event_loop().time()
                        }))
                    finally:
                        import os
                        os.unlink(temp_path)
                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "chunk_size": len(audio_chunk),
                    "buffer_size": len(audio_buffer)
                }))
            elif message["type"] == "end_stream":
                # If preserve_speakers, run diarization and chunked transcription
                if preserve_speakers and not diarization_done:
                    diarization_done = True
                    # Diarization pass
                    engine = TranscriptionEngine(
                        engine="faster-whisper",
                        model_size=model_size,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        enable_speaker_diarization=True
                    )
                    diarization_result = engine.transcribe(temp_audio_file.name, language=language)
                    speaker_segments = diarization_result.get('speaker_segments', [])
                    speaker_timeline = []
                    for segment in speaker_segments:
                        speaker_timeline.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'speaker': segment['speaker']
                        })
                    # Chunked transcription pass
                    audio = np.array(audio_buffer, dtype=np.float32)
                    duration = len(audio) / sample_rate
                    chunk_size = chunk_duration
                    for start_time_chunk in range(0, int(duration), int(chunk_size)):
                        end_time_chunk = min(start_time_chunk + chunk_size, int(duration))
                        start_sample = int(start_time_chunk * sample_rate)
                        end_sample = int(end_time_chunk * sample_rate)
                        audio_chunk = audio[start_sample:end_sample]
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            sf.write(temp_file.name, audio_chunk, sample_rate)
                            temp_path = temp_file.name
                        try:
                            chunk_engine = TranscriptionEngine(
                                engine="faster-whisper",
                                model_size=model_size,
                                device=device,
                                compute_type=compute_type,
                                cpu_threads=cpu_threads,
                                enable_speaker_diarization=False
                            )
                            result = chunk_engine.transcribe(temp_path, language=language)
                            chunk_segments = result.get('segments', [])
                            # Map speakers
                            mapped_segments = []
                            for segment in chunk_segments:
                                segment_start = segment['start'] + start_time_chunk
                                segment_end = segment['end'] + start_time_chunk
                                active_speaker = None
                                max_overlap = 0
                                for speaker_seg in speaker_timeline:
                                    overlap_start = max(segment_start, speaker_seg['start'])
                                    overlap_end = min(segment_end, speaker_seg['end'])
                                    overlap = max(0, overlap_end - overlap_start)
                                    if overlap > max_overlap:
                                        max_overlap = overlap
                                        active_speaker = speaker_seg['speaker']
                                mapped_segments.append({
                                    'start': segment_start,
                                    'end': segment_end,
                                    'text': segment.get('text', ''),
                                    'speaker': active_speaker or 'Unknown',
                                    'confidence': segment.get('avg_logprob', 0.0)
                                })
                            # Send mapped segments
                            for mapped in mapped_segments:
                                await websocket.send_text(json.dumps({
                                    "type": "transcription",
                                    "text": f"[{mapped['speaker']}] {mapped['text']}",
                                    "confidence": mapped['confidence'],
                                    "timestamp": asyncio.get_event_loop().time()
                                }))
                        finally:
                            import os
                            os.unlink(temp_path)
                    await websocket.send_text(json.dumps({
                        "type": "completed",
                        "message": "Streaming transcription with speaker preservation completed"
                    }))
                else:
                    # Process remaining audio in buffer (single speaker or default)
                    if audio_buffer:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            sf.write(temp_file.name, audio_buffer, sample_rate)
                            temp_path = temp_file.name
                        try:
                            engine = TranscriptionEngine(
                                engine="faster-whisper",
                                model_size=model_size,
                                device=device,
                                compute_type=compute_type,
                                cpu_threads=cpu_threads,
                                enable_speaker_diarization=not single_speaker
                            )
                            result = engine.transcribe(temp_path, language=language)
                            text = result.get("text", "").strip()
                            await websocket.send_text(json.dumps({
                                "type": "transcription",
                                "text": text,
                                "confidence": result.get("confidence", 0.0),
                                "timestamp": asyncio.get_event_loop().time(),
                                "final": True
                            }))
                        finally:
                            import os
                            os.unlink(temp_path)
                    await websocket.send_text(json.dumps({
                        "type": "completed",
                        "message": "Streaming transcription completed"
                    }))
                    break
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": asyncio.get_event_loop().time()
                }))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            }))
        except:
            pass 

 