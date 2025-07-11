#!/usr/bin/env python3
"""
Quick performance test for transcription platform
Tests both UIs and short audio file transcription
"""

import requests
import time
import os
import subprocess
import sys
from pathlib import Path

def test_ui_health(url, name):
    """Test if UI is running and healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} is running at {url}")
            return True
        else:
            print(f"❌ {name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name} not accessible: {e}")
        return False

def test_engine_info(url, name):
    """Test engine info endpoint"""
    try:
        response = requests.get(f"{url}/api/engine/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name} engine info: {data.get('engines', [])}")
            return True
        else:
            print(f"❌ {name} engine info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name} engine info error: {e}")
        return False

def test_short_audio_transcription(url, name):
    """Test transcription with short audio file"""
    audio_file = Path("examples/sample_audio")
    if not audio_file.exists():
        print(f"❌ Sample audio directory not found: {audio_file}")
        return False
    
    # Find first audio file
    audio_files = list(audio_file.glob("*.wav")) + list(audio_file.glob("*.mp3")) + list(audio_file.glob("*.m4a"))
    if not audio_files:
        print(f"❌ No audio files found in {audio_file}")
        return False
    
    test_file = audio_files[0]
    print(f"🎵 Testing with: {test_file}")
    
    try:
        # Upload file
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'audio/wav')}
            data = {
                'language': 'en',
                'engine': 'faster-whisper',
                'model': 'base'
            }
            
            print(f"📤 Uploading to {name}...")
            start_time = time.time()
            response = requests.post(f"{url}/api/upload", files=files, data=data, timeout=30)
            upload_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload successful in {upload_time:.2f}s")
                
                # Get job ID and monitor progress
                if 'job_id' in result:
                    job_id = result['job_id']
                    print(f"🔄 Monitoring job {job_id}...")
                    
                    # Monitor for up to 60 seconds
                    for i in range(60):
                        time.sleep(1)
                        try:
                            job_response = requests.get(f"{url}/api/jobs/{job_id}", timeout=5)
                            if job_response.status_code == 200:
                                job_data = job_response.json()
                                status = job_data.get('status', 'unknown')
                                progress = job_data.get('progress', 0)
                                
                                if status == 'completed':
                                    print(f"✅ Transcription completed in {i+1}s")
                                    if 'transcriptions' in job_data and job_data['transcriptions']:
                                        transcript = job_data['transcriptions'][0]
                                        text = transcript.get('text', '')[:100]
                                        print(f"📝 Preview: {text}...")
                                    return True
                                elif status == 'failed':
                                    print(f"❌ Transcription failed: {job_data.get('error_message', 'Unknown error')}")
                                    return False
                                else:
                                    print(f"🔄 Status: {status}, Progress: {progress}%")
                            else:
                                print(f"⚠️  Job status check failed: {job_response.status_code}")
                        except requests.exceptions.RequestException as e:
                            print(f"⚠️  Job status check error: {e}")
                    
                    print(f"⏰ Transcription timed out after 60s")
                    return False
                else:
                    print(f"❌ No job_id in response: {result}")
                    return False
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Transcription test error: {e}")
        return False

def main():
    print("🚀 Quick Performance Test for Transcription Platform")
    print("=" * 60)
    
    # Test Basic UI
    print("\n📋 Testing Basic UI (port 8000)...")
    basic_ui_ok = test_ui_health("http://localhost:8000", "Basic UI")
    if basic_ui_ok:
        test_engine_info("http://localhost:8000", "Basic UI")
        test_short_audio_transcription("http://localhost:8000", "Basic UI")
    
    # Test Scripflow UI
    print("\n📋 Testing Scripflow UI (port 8010)...")
    scripflow_ok = test_ui_health("http://localhost:8010", "Scripflow UI")
    if scripflow_ok:
        test_engine_info("http://localhost:8010", "Scripflow UI")
        test_short_audio_transcription("http://localhost:8010", "Scripflow UI")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"Basic UI: {'✅ Running' if basic_ui_ok else '❌ Not running'}")
    print(f"Scripflow UI: {'✅ Running' if scripflow_ok else '❌ Not running'}")
    
    if not basic_ui_ok and not scripflow_ok:
        print("\n💡 To start the UIs, run:")
        print("   python start_basic_ui.py")
        print("   python start_scripflow.py")

if __name__ == "__main__":
    main() 