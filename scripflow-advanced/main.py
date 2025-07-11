#!/usr/bin/env python3
"""
Scripflow Transcription Platform - Main Entry Point
Runs on port 8001 with separate UI and backend
"""

import uvicorn
import os
import sys
import platform
import subprocess

# Windows multiprocessing fix
if platform.system() == 'Windows':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Disable colorama to prevent import issues
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"

# Configuration will be loaded by the backend process

# This file just launches the backend server
# The actual FastAPI app is in backend/src/main.py

def main():
    print("🚀 Starting Scripflow UI on port 8001...")
    print("📁 Serving from: scripflow-advanced/")
    print("🔗 URL: http://localhost:8001")
    print("=" * 50)
    
    try:
        # Load configuration (including HF token)
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from config import config
            print("✅ Configuration loaded")
        except ImportError:
            print("⚠️  Config module not found, continuing without token")
        
        # Start the Scripflow backend server directly
        backend_src_dir = os.path.join(os.path.dirname(__file__), 'backend', 'src')
        subprocess.run([
            sys.executable, os.path.join(backend_src_dir, "main.py")
        ], check=True)
    except KeyboardInterrupt:
        print("\n⚠️  Scripflow UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting Scripflow UI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 