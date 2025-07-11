#!/usr/bin/env python3
"""
Complete Colab setup script for transcription platform.
This script handles setup, HuggingFace token, and provides public URL.
"""

import os
import sys
import subprocess
import time
import requests
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

def setup_hf_token():
    """Setup HuggingFace token interactively."""
    print("\n🔑 HuggingFace Token Setup")
    print("=" * 40)
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
        return True
    else:
        print("⚠️  Skipped HuggingFace token setup")
        return True

def get_public_url():
    """Get the public URL for the server."""
    print("\n🌐 Getting Public URL...")
    print("=" * 40)
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            
            # Get Colab session info
            colab_session = os.environ.get('COLAB_SESSION_ID', 'unknown')
            
            print("\n🌐 Your Public URLs:")
            print("1. Internal URL: http://127.0.0.1:8000")
            print("2. Colab URL: https://[session-id]-8000.colab.research.google.com")
            print("\n📱 To get the exact public URL:")
            print("   - Go to Runtime → Manage sessions")
            print("   - Click 'Connect to localhost:8000'")
            print("   - Or use the URL format above")
            
            return True
        else:
            print("❌ Server not responding properly")
            return False
    except Exception as e:
        print(f"❌ Server not running: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Complete Transcription Platform Setup for Colab")
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
        print("⚠️  System update failed, continuing...")
    
    if not run_command("apt-get install -y ffmpeg", "Installing FFmpeg"):
        print("⚠️  FFmpeg installation failed, continuing...")
    
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
        print("⚠️  Model download failed, continuing...")
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    if not run_command("python -c \"from src.database.init_db import init_database; init_database()\"", "Database initialization"):
        print("⚠️  Database initialization failed, continuing...")
    
    # Setup HuggingFace token
    setup_hf_token()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Upload an audio file using: files.upload()")
    print("2. Move file to uploads directory")
    print("3. Start the server with: !python src/main.py")
    print("4. Get public URL using the instructions above")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 