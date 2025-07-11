#!/usr/bin/env python3
"""
Google Colab setup script for the transcription platform.
This script installs all dependencies and sets up the environment for Google Colab.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and print status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Transcription Platform for Google Colab")
    print("=" * 60)
    
    # Check if running in Colab
    is_colab = 'COLAB_GPU' in os.environ
    if is_colab:
        print("✅ Running in Google Colab environment")
    else:
        print("⚠️  Not running in Google Colab - some features may not work optimally")
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    if not run_command("apt-get update", "Updating package list"):
        return False
    
    if not run_command("apt-get install -y ffmpeg", "Installing FFmpeg"):
        return False
    
    # Install Python dependencies
    print("\n🐍 Installing Python dependencies...")
    if not run_command("pip install -r requirements_local.txt", "Installing Python packages"):
        return False
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ['uploads', 'models', 'logs', 'transcript_output']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")
    
    # Download models
    print("\n🤖 Downloading Whisper models...")
    if not run_command("python scripts/download_models.py", "Downloading models"):
        return False
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    if not run_command("python -c \"from src.database.init_db import init_database; init_database()\"", "Database initialization"):
        return False
    
    # Set up HuggingFace token if provided
    print("\n🔑 HuggingFace Token Setup")
    print("For speaker diarization, you'll need a HuggingFace token.")
    print("Get one from: https://huggingface.co/settings/tokens")
    print()
    
    token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if token:
        if not token.startswith('hf_'):
            print("❌ Token should start with 'hf_'")
            return False
        
        # Save token to config
        config_content = f'''{{
  "hf_token": "{token}"
}}'''
        with open('config.json', 'w') as f:
            f.write(config_content)
        print("✅ HuggingFace token saved")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Upload an audio file to the 'uploads' directory")
    print("2. Run the transcription: python src/main.py")
    print("3. Access the web interface at the provided URL")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 