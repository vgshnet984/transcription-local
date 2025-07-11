import os
import sys
import platform
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Setup logging first - IMPORTANT: Do this before any other imports
from logging_config import setup_minimal_logging, suppress_model_loading_logs
setup_minimal_logging()
suppress_model_loading_logs()

# Suppress SQLAlchemy logs
logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.pool').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.ERROR)

# Now import other modules
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import traceback

from config import settings
from database.init_db import init_database, get_db
from database.models import AudioFile, TranscriptionJob, Transcription
from transcription.engine import TranscriptionEngine
from loguru import logger

def create_app():
    app = Flask(__name__, 
                template_folder='../templates',  # Point to the correct templates directory
                static_folder='../static')       # Point to the correct static directory
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    
    # Initialize database
    init_database()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/engine-status')
    def engine_status():
        """Get engine status and available options."""
        try:
            # Create a minimal engine instance to get info
            engine = TranscriptionEngine(suppress_logs=True)
            model_info = engine.get_model_info()
            
            # Get available options from settings (optimized for performance)
            available_engines = ["whisper", "faster-whisper", "whisperx"]
            available_models = ["tiny", "base", "small", "medium", "large", "large-v3"]
            available_vad_methods = ["none", "simple", "webrtcvad", "silero"]
            
            return jsonify({
                "status": "ready",
                "available_engines": available_engines,
                "available_models": available_models,
                "available_vad_methods": available_vad_methods,
                "model_info": model_info
            })
        except Exception as e:
            logger.error(f"Engine status error: {e}")
            return jsonify({
                "status": "error",
                "error": str(e)
            }), 500
    
    @app.route('/api/upload', methods=['POST'])
    def upload_audio():
        """Upload audio file."""
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Validate file type
            allowed_extensions = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
            filename = file.filename or "unknown"
            if not filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
                return jsonify({"error": "Invalid file type"}), 400
            
            # Save file
            safe_filename = secure_filename(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{timestamp}_{safe_filename}"
            file_path = os.path.join(settings.upload_dir, final_filename)
            
            # Ensure upload directory exists
            os.makedirs(settings.upload_dir, exist_ok=True)
            
            file.save(file_path)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            
            # Save to database
            db = next(get_db())
            try:
                audio_file = AudioFile(
                    filename=final_filename,
                    path=file_path,
                    size=file_size,
                    format=filename.split('.')[-1].lower(),
                    status="uploaded"
                )
                db.add(audio_file)
                db.commit()
                db.refresh(audio_file)
                
                return jsonify({
                    "id": audio_file.id,
                    "filename": final_filename,
                    "size": file_size,
                    "status": "uploaded"
                })
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({"error": f"Upload failed: {str(e)}"}), 500
    
    @app.route('/api/transcribe', methods=['POST'])
    def transcribe_audio():
        """Transcribe uploaded audio file."""
        try:
            data = request.get_json()
            file_id = data.get('file_id')
            
            if not file_id:
                return jsonify({"error": "No file ID provided"}), 400
            
            # Get file from database
            db = next(get_db())
            try:
                audio_file = db.query(AudioFile).filter(AudioFile.id == file_id).first()
                if not audio_file:
                    return jsonify({"error": "File not found"}), 404
                
                # Get transcription options
                engine = data.get('engine', 'whisper')
                model_size = data.get('model_size', 'base')
                language = data.get('language', 'auto')
                vad_method = data.get('vad_method', 'simple')
                enable_speaker_diarization = data.get('enable_speaker_diarization', False)
                no_filtering = data.get('no_filtering', False)
                suppress_logs = data.get('suppress_logs', True)
                
                # Create transcription job
                config_dict = {
                    "engine": engine,
                    "model_size": model_size,
                    "language": language,
                    "vad_method": vad_method,
                    "enable_speaker_diarization": enable_speaker_diarization,
                    "no_filtering": no_filtering,
                    "suppress_logs": suppress_logs
                }
                config_json = json.dumps(config_dict)
                logger.info(f"Creating job with config: {config_json}")
                
                job = TranscriptionJob(
                    audio_file_id=file_id,
                    status="processing",
                    progress=0,
                    config=config_json
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                
                # Extract file path before closing session
                audio_file_path = str(audio_file.path)
                job_id = job.id
                
                # Start transcription in background
                def transcribe_background():
                    try:
                        # Create engine with options
                        transcription_engine = TranscriptionEngine(
                            model_size=model_size,
                            device=settings.device,
                            engine=engine,
                            vad_method=vad_method,
                            enable_speaker_diarization=enable_speaker_diarization,
                            suppress_logs=suppress_logs
                        )
                        
                        # Transcribe using the file path
                        result = transcription_engine.transcribe(audio_file_path, language)
                        
                        # Save result
                        db = next(get_db())
                        try:
                            # Calculate additional metrics
                            text = result.get('text', '')
                            word_count = len(text.split()) if text else 0
                            character_count = len(text) if text else 0
                            processing_time = result.get('processing_time', 0.0)
                            words_per_second = word_count / processing_time if processing_time > 0 else 0
                            
                            # Get VAD results from logger if available
                            vad_results = getattr(transcription_engine.logger, 'run_data', {}).get('vad_results', {})
                            
                            # Get speaker information
                            speaker_segments = result.get('speaker_segments', [])
                            speaker_count = len(set(seg.get('speaker', '') for seg in speaker_segments)) if speaker_segments else 0
                            
                            # Get GPU information
                            import torch
                            gpu_used = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                            
                            transcription = Transcription(
                                job_id=job_id,
                                text=text,
                                segments_json=json.dumps(result.get('segments', [])),
                                confidence=result.get('confidence', 0.0),
                                processing_time=processing_time,
                                
                                # Quality analysis fields
                                engine_used=result.get('engine_used', engine),
                                model_size=result.get('model_size', model_size),
                                vad_method=result.get('vad_method', vad_method),
                                language_detected=result.get('language', language),
                                word_count=word_count,
                                character_count=character_count,
                                words_per_second=words_per_second,
                                
                                # VAD results
                                vad_segments_count=vad_results.get('segments_count', 0),
                                vad_total_speech_duration=vad_results.get('total_speech_duration', 0.0),
                                vad_processing_time=vad_results.get('processing_time', 0.0),
                                
                                # Speaker diarization
                                speaker_segments_json=json.dumps(speaker_segments),
                                speaker_count=speaker_count,
                                
                                # Performance metrics
                                gpu_used=gpu_used,
                                memory_usage_mb=0.0  # Could be enhanced with actual memory monitoring
                            )
                            db.add(transcription)
                            
                            # Update job status - need to query the job again
                            job_to_update = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
                            if job_to_update:
                                job_to_update.status = "completed"
                                job_to_update.progress = 100.0
                                job_to_update.completed_at = datetime.now()
                            
                            db.commit()
                            
                            logger.info(f"✅ Transcription saved to database: {word_count} words, {result.get('confidence', 0.0):.3f} confidence")
                            
                        finally:
                            db.close()
                            
                    except Exception as e:
                        logger.error(f"Transcription error: {e}")
                        db = next(get_db())
                        try:
                            job_to_update = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
                            if job_to_update:
                                job_to_update.status = "failed"
                                job_to_update.error_message = str(e)
                            db.commit()
                        finally:
                            db.close()
                
                # Start background thread
                import threading
                thread = threading.Thread(target=transcribe_background)
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    "job_id": job.id,
                    "status": "processing",
                    "message": "Transcription started"
                })
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    
    @app.route('/api/jobs/<int:job_id>/status')
    def job_status(job_id):
        """Get job status."""
        try:
            db = next(get_db())
            try:
                job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
                if not job:
                    return jsonify({"error": "Job not found"}), 404
                
                return jsonify({
                    "job_id": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "error_message": job.error_message,
                    "created_at": job.created_at.isoformat() if job.created_at is not None else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at is not None else None
                })
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Job status error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/jobs/<int:job_id>/result')
    def job_result(job_id):
        """Get transcription result."""
        try:
            db = next(get_db())
            try:
                transcription = db.query(Transcription).filter(Transcription.job_id == job_id).first()
                if not transcription:
                    return jsonify({"error": "Transcription not found"}), 404
                
                segments_json = transcription.segments_json
                segments = json.loads(segments_json) if segments_json else []
                
                return jsonify({
                    "text": transcription.text,
                    "segments": segments,
                    "confidence": transcription.confidence,
                    "processing_time": transcription.processing_time
                })
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Job result error: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    logger.info("🚀 Starting transcription platform...")
    logger.info(f"📁 Upload directory: {settings.upload_dir}")
    logger.info(f"🔧 Database: {settings.database_url}")
    
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=False)