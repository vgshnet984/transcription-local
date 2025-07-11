#!/usr/bin/env python3
"""Start the Basic UI transcription platform."""

import os
import sys
import subprocess
from pathlib import Path

# Add cuDNN path if available
cudnn_path = "C:\\cudnn\\bin"
if os.path.exists(cudnn_path):
    os.environ["PATH"] = cudnn_path + ";" + os.environ.get("PATH", "")
    print(f"✅ Added cuDNN PATH: {cudnn_path}")

print("🚀 Starting Basic UI on port 8000...")
print("📁 Serving from: templates/")
print("🔗 URL: http://localhost:8000")
print("=" * 50)

try:
    # Change to src directory and run main.py
    src_path = Path(__file__).parent / "src"
    os.chdir(src_path)
    
    # Set PYTHONPATH to include src directory
    os.environ["PYTHONPATH"] = str(src_path)
    
    result = subprocess.run([sys.executable, "main.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Error starting Basic UI: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nStopped by user")
    sys.exit(0) 