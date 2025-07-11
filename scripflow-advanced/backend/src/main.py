import uvicorn
import os
import sys
import platform
import logging

# Aggressively suppress SQLAlchemy logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.orm").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.sql").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.event").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.base").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.Connection").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.Transaction").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.Result").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.CursorResult").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.Row").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.RowMapping").setLevel(logging.ERROR)

# Windows multiprocessing fix
if platform.system() == 'Windows':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from fastapi import FastAPI, UploadFile, Form, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from typing import Optional
from sqlalchemy.orm import Session

# Disable colorama to prevent import issues
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"

from config import settings
from api.routes import router as api_router, get_db, upload_audio


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        colorize=False  # Disable colorization
    )
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        colorize=False  # Disable colorization
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="Scripflow - Advanced Transcription Platform",
        description="Advanced transcription platform with enhanced UI",
        version="1.0.0",
        debug=settings.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Mount uploads directory for audio file access
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "scripflow-advanced"}
    
    # Mount Scripflow static files (subdirectories) using absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    scripflow_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    app.mount("/scripflow/js", StaticFiles(directory=os.path.join(scripflow_root, "js")), name="scripflow-js")
    app.mount("/scripflow/css", StaticFiles(directory=os.path.join(scripflow_root, "css")), name="scripflow-css")
    app.mount("/scripflow/assets", StaticFiles(directory=os.path.join(scripflow_root, "assets")), name="scripflow-assets")
    
    # Scripflow main interface (serve Scripflow's index.html at root)
    @app.get("/")
    async def scripflow_root_index():
        return FileResponse(os.path.join(scripflow_root, "index.html"))
    
    # Mount uploads directory for audio file access (if Scripflow needs it)
    uploads_dir = os.path.join(project_root, 'uploads')
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
    
    # Include API routes
    app.include_router(api_router, prefix="/api")
    
    # Legacy upload route for compatibility
    @app.post("/upload")
    async def legacy_upload(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        config: Optional[str] = Form(None),
        db: Session = Depends(get_db)
    ):
        """Legacy upload endpoint that includes transcription."""
        try:
            # Parse configuration if provided
            transcription_config = {}
            
            # Parse config JSON if provided
            if config:
                try:
                    import json
                    config_dict = json.loads(config)
                    transcription_config.update(config_dict)
                    logger.info(f"Using configuration: {transcription_config}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse config JSON: {e}")
            
            # Fallback to language parameter if no config
            if language and "language" not in transcription_config:
                transcription_config["language"] = language
            
            # Save uploaded file
            upload_result = await upload_audio(file, db)
            if not upload_result:
                logger.error("Failed to save uploaded file in legacy upload endpoint")
                return {"error": "Failed to save uploaded file"}
            audio_file_id = upload_result["id"]
            
            # Start transcription
            try:
                from api.routes import start_transcription
                transcription_id = await start_transcription(audio_file_id, transcription_config, db)
                if not isinstance(transcription_id, int):
                    logger.error(f"start_transcription did not return an int: {transcription_id}")
                    return {"error": "Failed to create transcription"}
                return {"transcription_id": transcription_id}
            except Exception as e:
                logger.error(f"Exception in legacy upload (transcription): {e}")
                return {"error": str(e)}
        except Exception as e:
            logger.error(f"Exception in legacy upload: {e}")
            return {"error": str(e)}
    
    # Legacy transcribe route for compatibility
    @app.post("/transcribe")
    async def legacy_transcribe(
        file: Optional[UploadFile] = File(None),
        language: Optional[str] = Form(None),
        db: Session = Depends(get_db)
    ):
        if not file:
            raise HTTPException(status_code=422, detail="No file provided")
        from api.routes import upload_audio
        return await upload_audio(file=file, db=db)
    
    # Legacy jobs route for compatibility
    @app.get("/jobs")
    def legacy_jobs(db: Session = Depends(get_db)):
        from api.routes import list_jobs
        return list_jobs(db=db)
    
    return app


app = create_app()


def main():
    """Main entry point for the application."""
    logger.info("Starting transcription platform...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database URL: {settings.database_url}")
    logger.info(f"Upload directory: {settings.upload_dir}")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )


if __name__ == "__main__":
    main() 