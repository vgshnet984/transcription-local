#!/usr/bin/env python3
"""
Enhanced Test script for Scripflow (port 8001)
Tests upload, transcription, and file output with detailed logging
"""

import requests
import time
import os
import json
import datetime
from pathlib import Path

def log_message(message, level="INFO"):
    """Log message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def test_scripflow_enhanced():
    """Test the Scripflow functionality with enhanced logging."""
    base_url = "http://localhost:8001"
    sample_audio = "examples/sample_audio/tes1_first_35s.wav"
    
    log_message("=" * 60, "TEST")
    log_message("Testing Scripflow (port 8001) - Enhanced Version", "TEST")
    log_message("=" * 60, "TEST")
    
    # Check if server is running
    log_message("Checking Scripflow server status...", "CHECK")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            log_message("Scripflow server not responding", "ERROR")
            return False
        log_message("Scripflow server is running and healthy", "SUCCESS")
    except requests.exceptions.RequestException:
        log_message("Scripflow server not accessible", "ERROR")
        return False
    
    # Check if sample audio exists
    if not os.path.exists(sample_audio):
        log_message(f"Sample audio not found: {sample_audio}", "ERROR")
        return False
    log_message(f"Sample audio found: {sample_audio}", "SUCCESS")
    
    # Upload file
    log_message("Starting file upload process...", "UPLOAD")
    try:
        with open(sample_audio, 'rb') as f:
            files = {'file': (os.path.basename(sample_audio), f, 'audio/wav')}
            log_message("Sending file to server...", "UPLOAD")
            response = requests.post(f"{base_url}/api/upload", files=files)
        
        if response.status_code != 200:
            log_message(f"Upload failed with status: {response.status_code}", "ERROR")
            return False
        
        upload_result = response.json()
        file_id = upload_result.get('id')
        if not file_id:
            log_message("No file ID returned from upload", "ERROR")
            return False
        
        log_message(f"File uploaded successfully, ID: {file_id}", "SUCCESS")
        
    except Exception as e:
        log_message(f"Upload error: {e}", "ERROR")
        return False
    
    # Start transcription
    log_message("Starting transcription process...", "TRANSCRIBE")
    try:
        config = {
            "language": "en",
            "transcription_engine": "whisper",
            "whisper_model": "base"
        }
        
        log_message("Sending transcription request...", "TRANSCRIBE")
        response = requests.post(
            f"{base_url}/api/transcribe",
            json={"file_id": file_id, "config": config}
        )
        
        if response.status_code != 200:
            log_message(f"Transcription start failed: {response.status_code}", "ERROR")
            return False
        
        transcribe_result = response.json()
        job_id = transcribe_result.get('transcription_id')
        if not job_id:
            log_message("No job ID returned from transcription start", "ERROR")
            return False
        
        log_message(f"Transcription started, Job ID: {job_id}", "SUCCESS")
        
    except Exception as e:
        log_message(f"Transcription start error: {e}", "ERROR")
        return False
    
    # Wait for transcription to complete with detailed progress
    log_message("Monitoring transcription progress...", "PROGRESS")
    max_wait = 120  # 2 minutes
    wait_time = 0
    last_progress = -1
    
    while wait_time < max_wait:
        try:
            response = requests.get(f"{base_url}/api/jobs/{job_id}")
            if response.status_code == 200:
                job_data = response.json()
                status = job_data.get('status')
                progress = job_data.get('progress', 0)
                
                # Only log progress changes
                if progress != last_progress:
                    log_message(f"Job Status: {status}, Progress: {progress:.1f}%", "PROGRESS")
                    last_progress = progress
                
                if status == 'completed':
                    log_message("Transcription completed successfully!", "SUCCESS")
                    break
                elif status == 'failed':
                    error_msg = job_data.get('error_message', 'Unknown error')
                    log_message(f"Transcription failed: {error_msg}", "ERROR")
                    return False
                    
            time.sleep(3)  # Check every 3 seconds
            wait_time += 3
            
        except Exception as e:
            log_message(f"Error checking job status: {e}", "ERROR")
            return False
    
    if wait_time >= max_wait:
        log_message("Transcription timed out after 2 minutes", "ERROR")
        return False
    
    # Check for transcript file
    log_message("Checking for transcript file output...", "OUTPUT")
    transcript_file = f"transcripts/transcription_{job_id}.txt"
    if os.path.exists(transcript_file):
        log_message(f"Transcript file created: {transcript_file}", "SUCCESS")
        
        # Read and display file info
        file_size = os.path.getsize(transcript_file)
        log_message(f"File size: {file_size} bytes", "INFO")
        
        # Read and display first few lines
        try:
            with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')[:8]
                log_message("Transcript preview:", "PREVIEW")
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        log_message(f"  {i:2d}: {line.strip()}", "PREVIEW")
        except Exception as e:
            log_message(f"Error reading transcript file: {e}", "WARNING")
    else:
        log_message(f"Transcript file not found: {transcript_file}", "ERROR")
        return False
    
    # Test download endpoint
    log_message("Testing download functionality...", "DOWNLOAD")
    try:
        response = requests.get(f"{base_url}/api/transcriptions/{job_id}/download")
        if response.status_code == 200:
            log_message("Download endpoint working correctly", "SUCCESS")
        else:
            log_message(f"Download endpoint failed: {response.status_code}", "WARNING")
    except Exception as e:
        log_message(f"Download test error: {e}", "WARNING")
    
    log_message("=" * 60, "TEST")
    log_message("Scripflow test completed successfully!", "SUCCESS")
    log_message("=" * 60, "TEST")
    return True

if __name__ == "__main__":
    success = test_scripflow_enhanced()
    exit(0 if success else 1) 