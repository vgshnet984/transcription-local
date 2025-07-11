#!/usr/bin/env python3
"""
Download Whisper models for local transcription platform.
"""

import os
import sys
from pathlib import Path
import whisper
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings


def download_whisper_model(model_name: str = None):
    """Download Whisper model."""
    if model_name is None:
        model_name = settings.whisper_model
    
    try:
        logger.info(f"Downloading Whisper model: {model_name}")
        
        # Create models directory
        models_dir = Path(settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model (this will cache it in the models directory)
        model = whisper.load_model(model_name, download_root=str(models_dir))
        
        logger.info(f"✅ Whisper model '{model_name}' downloaded successfully")
        logger.info(f"📁 Model location: {models_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download Whisper model: {e}")
        return False


def list_available_models():
    """List available Whisper models."""
    models = ["tiny", "base", "small", "medium", "large", "large-v2"]
    
    print("Available Whisper models:")
    for model in models:
        print(f"  - {model}")
    
    return models


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Whisper models")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Whisper model to download (default: from settings)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    # Download specified model or default
    success = download_whisper_model(args.model)
    
    if success:
        print("\n🎉 Model download completed!")
        print("You can now run the transcription platform.")
    else:
        print("\n❌ Model download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 