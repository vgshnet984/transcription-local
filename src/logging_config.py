"""Minimal logging configuration for transcription platform."""

import logging
import warnings
from loguru import logger

def setup_minimal_logging():
    """Configure logging to show only essential information."""
    
    # Suppress all verbose logs
    logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.ERROR)
    logging.getLogger('sqlalchemy.dialects').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('librosa').setLevel(logging.ERROR)
    logging.getLogger('numpy').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    logging.getLogger('tokenizers').setLevel(logging.ERROR)
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*speechbrain.*")
    
    # Configure loguru for clean output
    logger.remove()
    
    # Console output - minimal format
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>\n",
        level="INFO",
        filter=lambda record: record["level"].no >= 20 and "webrtcvad" not in record["message"]
    )
    
    # File output - detailed for debugging
    logger.add(
        "logs/detailed.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )

def suppress_model_loading_logs():
    """Suppress verbose model loading logs."""
    # Suppress specific model loading messages
    original_info = logger.info
    
    def filtered_info(message):
        # Filter out verbose model loading messages
        if any(keyword in message.lower() for keyword in [
            "available", "not available", "loading", "loaded successfully",
            "falling back", "disabled due to", "speechbrain", "webrtcvad"
        ]):
            return
        original_info(message)
    
    logger.info = filtered_info 