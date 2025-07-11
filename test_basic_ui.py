#!/usr/bin/env python3
"""
Test script for Basic UI (port 8000)
Tests upload, transcription, and file output
"""

import requests
import time
import os
import json
from pathlib import Path

def test_basic_ui():
    """Test the basic UI functionality."""
    base_url = "http://localhost:8000"
    sample_audio = "examples/sample_audio/tes1_first_35s"
    
    print("🧪 Testing Basic UI (port 8000)")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Basic UI server not responding")
            return False
        print("✅ Basic UI server is running")
    except requests.exceptions.RequestException:
        print("❌ Basic UI server not accessible")
        return False
    
    # Check if sample audio exists
    if not os.path.exists(sample_audio):
        print(f"❌ Sample audio not found: {sample_audio}")
        return False
    print(f"✅ Sample audio found: {sample_audio}")
    
    # Upload file
    print("\n📤 Uploading audio file...")
    try:
        with open(sample_audio, 'rb') as f:
            files = {'file': (os.path.basename(sample_audio), f, 'audio/wav')}
            response = requests.post(f"{base_url}/api/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            return False
        
        upload_result = response.json()
        file_id = upload_result.get('id')
        if not file_id:
            print("❌ No file ID returned from upload")
            return False
        
        print(f"✅ File uploaded successfully, ID: {file_id}")
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    # Start transcription
    print("\n🎤 Starting transcription...")
    try:
        config = {
            "language": "en",
            "transcription_engine": "whisper",
            "whisper_model": "base"
        }
        
        response = requests.post(
            f"{base_url}/api/transcribe",
            json={"file_id": file_id, "config": config}
        )
        
        if response.status_code != 200:
            print(f"❌ Transcription start failed: {response.status_code}")
            return False
        
        transcribe_result = response.json()
        job_id = transcribe_result.get('transcription_id')
        if not job_id:
            print("❌ No job ID returned from transcription start")
            return False
        
        print(f"✅ Transcription started, Job ID: {job_id}")
        
    except Exception as e:
        print(f"❌ Transcription start error: {e}")
        return False
    
    # Wait for transcription to complete
    print("\n⏳ Waiting for transcription to complete...")
    max_wait = 60  # 60 seconds
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            response = requests.get(f"{base_url}/api/jobs/{job_id}")
            if response.status_code == 200:
                job_data = response.json()
                status = job_data.get('status')
                progress = job_data.get('progress', 0)
                
                print(f"   Status: {status}, Progress: {progress:.1f}%")
                
                if status == 'completed':
                    print("✅ Transcription completed!")
                    break
                elif status == 'failed':
                    print(f"❌ Transcription failed: {job_data.get('error_message', 'Unknown error')}")
                    return False
                    
            time.sleep(2)
            wait_time += 2
            
        except Exception as e:
            print(f"❌ Error checking job status: {e}")
            return False
    
    if wait_time >= max_wait:
        print("❌ Transcription timed out")
        return False
    
    # Check for transcript file
    print("\n📁 Checking for transcript file...")
    transcript_file = f"transcripts/transcription_{job_id}.txt"
    if os.path.exists(transcript_file):
        print(f"✅ Transcript file created: {transcript_file}")
        
        # Read and display first few lines
        with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')[:5]
            print("   Preview:")
            for line in lines:
                if line.strip():
                    print(f"   {line}")
    else:
        print(f"❌ Transcript file not found: {transcript_file}")
        return False
    
    print("\n🎉 Basic UI test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_basic_ui()
    exit(0 if success else 1) 