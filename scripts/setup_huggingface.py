#!/usr/bin/env python3
"""
Setup script for Hugging Face speaker diarization models.
This script helps download and configure the required models.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_pyannote_installed():
    """Check if pyannote.audio is installed."""
    try:
        import pyannote.audio
        print("✓ pyannote.audio is installed")
        return True
    except ImportError:
        print("✗ pyannote.audio is not installed")
        return False

def install_pyannote():
    """Install pyannote.audio."""
    print("Installing pyannote.audio...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyannote.audio"], check=True)
        print("✓ pyannote.audio installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install pyannote.audio: {e}")
        return False

def check_hf_token():
    """Check if HF_TOKEN is set."""
    token = os.getenv("HF_TOKEN")
    if token:
        print("✓ HF_TOKEN is set")
        return True
    else:
        print("✗ HF_TOKEN is not set")
        return False

def get_hf_token():
    """Get Hugging Face token from user."""
    print("\nTo use speaker diarization, you need a Hugging Face token.")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is sufficient)")
    print("3. Copy the token")
    
    token = input("\nEnter your Hugging Face token: ").strip()
    if token:
        # Save to .env file
        env_file = Path(".env")
        with open(env_file, "w") as f:
            f.write(f"HF_TOKEN={token}\n")
        print(f"✓ Token saved to {env_file}")
        
        # Set environment variable for current session
        os.environ["HF_TOKEN"] = token
        return True
    else:
        print("✗ No token provided")
        return False

def accept_model_licenses():
    """Guide user to accept model licenses."""
    print("\nYou need to accept the model licenses on Hugging Face:")
    print("1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("   - Click 'Accept' on the license agreement")
    print("2. Go to https://huggingface.co/pyannote/voice-activity-detection-3.1")
    print("   - Click 'Accept' on the license agreement")
    
    input("\nPress Enter after accepting both licenses...")

def test_diarization():
    """Test the diarization setup."""
    print("\nTesting speaker diarization setup...")
    
    try:
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("✗ HF_TOKEN not set")
            return False
        
        print("Loading speaker diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        print("✓ Speaker diarization pipeline loaded successfully")
        
        print("Loading voice activity detection pipeline...")
        vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection-3.1",
            use_auth_token=token
        )
        print("✓ Voice activity detection pipeline loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("Hugging Face Speaker Diarization Setup")
    print("=" * 40)
    
    # Check pyannote installation
    if not check_pyannote_installed():
        if not install_pyannote():
            print("Failed to install pyannote.audio. Please install manually:")
            print("pip install pyannote.audio")
            return False
    
    # Check HF token
    if not check_hf_token():
        if not get_hf_token():
            print("Failed to get HF_TOKEN")
            return False
    
    # Guide user to accept licenses
    accept_model_licenses()
    
    # Test the setup
    if test_diarization():
        print("\n✓ Setup completed successfully!")
        print("\nYou can now use speaker diarization in your transcription platform.")
        print("The models will be downloaded automatically on first use.")
        return True
    else:
        print("\n✗ Setup failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 