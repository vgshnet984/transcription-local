#!/usr/bin/env python3
"""
Launch script for Scripflow (port 8010)
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Scripflow on port 8010...")
    print("📁 Serving from: scripflow-advanced/")
    print("🔗 URL: http://localhost:8010")
    print("=" * 50)
    
    try:
        # Load configuration (including HF token)
        try:
            from config import config
            print("✅ Configuration loaded")
        except ImportError:
            print("⚠️  Config module not found, continuing without token")
        
        # Set up environment with correct PYTHONPATH
        env = os.environ.copy()
        backend_src_dir = os.path.join(os.path.dirname(__file__), "scripflow-advanced", "backend", "src")
        env['PYTHONPATH'] = backend_src_dir + os.pathsep + env.get('PYTHONPATH', '')
        
        # Ensure C:\cudnn is in PATH for CUDA DLLs
        cudnn_path = r'C:\cudnn\bin'
        if cudnn_path not in env['PATH']:
            env['PATH'] = cudnn_path + os.pathsep + env['PATH']
        
        # Start the scripflow server from project root
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", "8010", "--reload",
            "--app-dir", backend_src_dir
        ], check=True, env=env)
    except KeyboardInterrupt:
        print("\n⚠️  Scripflow stopped by user")
    except Exception as e:
        print(f"❌ Error starting Scripflow: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 