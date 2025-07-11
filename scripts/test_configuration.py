#!/usr/bin/env python3
"""
Test script to verify configuration options and available features.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings
from loguru import logger

def test_configuration():
    """Test current configuration and available features."""
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)
    
    print(f"Current Configuration:")
    print(f"  Whisper Model: {settings.whisper_model}")
    print(f"  Device: {settings.device}")
    print(f"  Language: {settings.language}")
    print(f"  Transcription Engine: {settings.transcription_engine}")
    print(f"  VAD Method: {settings.vad_method}")
    print(f"  Enable VAD: {settings.enable_vad}")
    print(f"  Enable Audio Preprocessing: {settings.enable_audio_preprocessing}")
    print(f"  Enable Denoising: {settings.enable_denoising}")
    print(f"  Enable Normalization: {settings.enable_normalization}")
    
    print("\n" + "=" * 60)
    print("Available Features")
    print("=" * 60)
    
    # Check WhisperX availability
    try:
        import whisperx
        print("✅ WhisperX: Available")
        WHISPERX_AVAILABLE = True
    except ImportError:
        print("❌ WhisperX: Not available (install with: pip install git+https://github.com/m-bain/whisperx.git)")
        WHISPERX_AVAILABLE = False
    
    # Check webrtcvad availability
    try:
        import webrtcvad
        print("✅ WebRTC VAD: Available")
        WEBRTCVAD_AVAILABLE = True
    except ImportError:
        print("❌ WebRTC VAD: Not available (install with: pip install webrtcvad)")
        WEBRTCVAD_AVAILABLE = False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available ({torch.cuda.get_device_name(0)})")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("❌ CUDA: Not available (CPU only)")
    except ImportError:
        print("❌ PyTorch: Not available")
    
    print("\n" + "=" * 60)
    print("Recommended Configuration for Indian Accents")
    print("=" * 60)
    
    recommendations = []
    
    if WHISPERX_AVAILABLE:
        recommendations.append("Use WhisperX for enhanced transcription quality")
    else:
        recommendations.append("Install WhisperX: pip install git+https://github.com/m-bain/whisperx.git")
    
    if settings.whisper_model == "large-v3":
        recommendations.append("✅ Using large-v3 model (best for accents)")
    else:
        recommendations.append(f"Consider upgrading to large-v3 model (current: {settings.whisper_model})")
    
    if settings.device == "cuda":
        recommendations.append("✅ Using CUDA for faster processing")
    else:
        recommendations.append("Consider using CUDA for faster processing")
    
    if settings.enable_audio_preprocessing:
        recommendations.append("✅ Audio preprocessing enabled")
    else:
        recommendations.append("Enable audio preprocessing for better quality")
    
    if WEBRTCVAD_AVAILABLE:
        recommendations.append("✅ WebRTC VAD available for better voice detection")
    else:
        recommendations.append("Install WebRTC VAD: pip install webrtcvad")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 60)
    print("Configuration Options in UI")
    print("=" * 60)
    
    print("The web interface now includes configuration options:")
    print("  • Transcription Engine: Whisper vs WhisperX")
    print("  • Whisper Model: tiny, base, small, medium, large, large-v3")
    print("  • VAD Method: Simple, WebRTC VAD, Silero VAD")
    print("  • Device: CPU vs CUDA")
    
    print("\nTo experiment with different configurations:")
    print("  1. Open the web interface at http://127.0.0.1:8000")
    print("  2. Adjust the configuration panel settings")
    print("  3. Upload an audio file to test the configuration")
    print("  4. Compare results between different settings")

if __name__ == "__main__":
    test_configuration() 