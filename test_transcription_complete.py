import requests
import json
import time
import os
import sys
from pathlib import Path

def test_complete_transcription_flow():
    """Test the complete transcription flow with detailed logging."""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Complete Transcription Flow")
    print("=" * 50)
    
    # Test 1: Check server status
    print("\n1. Checking server status...")
    try:
        response = requests.get(f"{base_url}/engine-status", timeout=10)
        if response.status_code == 200:
            engine_data = response.json()
            print(f"✅ Server is running")
            print(f"   Available engines: {engine_data.get('available_engines', [])}")
            print(f"   Current engine: {engine_data.get('model_info', {}).get('current_engine', 'unknown')}")
        else:
            print(f"❌ Server status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Upload audio file
    print("\n2. Uploading audio file...")
    audio_file = "uploads/c942fd8d-c949-43eb-8246-5abe431073ec.m4a"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/api/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            upload_data = response.json()
            file_id = upload_data['id']
            print(f"✅ File uploaded successfully")
            print(f"   File ID: {file_id}")
            print(f"   Filename: {upload_data.get('filename', 'unknown')}")
            print(f"   Size: {upload_data.get('size', 0)} bytes")
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    # Test 3: Start transcription with specific options
    print("\n3. Starting transcription...")
    transcribe_data = {
        'file_id': file_id,
        'engine': 'faster-whisper',
        'model_size': 'large-v3',
        'language': 'en',
        'vad_method': 'simple',
        'enable_speaker_diarization': True,
        'no_filtering': False,
        'suppress_logs': True
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/transcribe", 
            json=transcribe_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data['job_id']
            print(f"✅ Transcription started successfully")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {job_data.get('status', 'unknown')}")
            print(f"   Message: {job_data.get('message', '')}")
        else:
            print(f"❌ Transcription start failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Transcription start error: {e}")
        return False
    
    # Test 4: Monitor transcription progress
    print("\n4. Monitoring transcription progress...")
    max_wait_time = 300  # 5 minutes max
    check_interval = 5   # Check every 5 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{base_url}/api/jobs/{job_id}/status", timeout=10)
            if response.status_code == 200:
                status_data = response.json()
                status = status_data['status']
                progress = status_data.get('progress', 0)
                error_msg = status_data.get('error_message')
                
                elapsed = time.time() - start_time
                print(f"   [{elapsed:.0f}s] Status: {status}, Progress: {progress}%")
                
                if error_msg:
                    print(f"   ❌ Error: {error_msg}")
                    return False
                
                if status == 'completed':
                    print(f"✅ Transcription completed in {elapsed:.1f} seconds!")
                    
                    # Test 5: Get transcription result
                    print("\n5. Retrieving transcription result...")
                    response = requests.get(f"{base_url}/api/jobs/{job_id}/result", timeout=10)
                    if response.status_code == 200:
                        result_data = response.json()
                        text = result_data.get('text', '')
                        confidence = result_data.get('confidence', 0)
                        processing_time = result_data.get('processing_time', 0)
                        
                        print(f"✅ Result retrieved successfully!")
                        print(f"   Text length: {len(text)} characters")
                        print(f"   Confidence: {confidence:.2f}")
                        print(f"   Processing time: {processing_time:.1f} seconds")
                        print(f"\n📝 Transcription Preview:")
                        print("-" * 40)
                        print(text[:500] + "..." if len(text) > 500 else text)
                        print("-" * 40)
                        
                        # Save result to file
                        output_file = f"transcript_output/complete_transcript_{job_id}.txt"
                        os.makedirs("transcript_output", exist_ok=True)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(f"Transcription Job ID: {job_id}\n")
                            f.write(f"File ID: {file_id}\n")
                            f.write(f"Engine: {transcribe_data['engine']}\n")
                            f.write(f"Model: {transcribe_data['model_size']}\n")
                            f.write(f"Language: {transcribe_data['language']}\n")
                            f.write(f"Confidence: {confidence:.2f}\n")
                            f.write(f"Processing Time: {processing_time:.1f}s\n")
                            f.write(f"Total Time: {elapsed:.1f}s\n")
                            f.write("-" * 50 + "\n")
                            f.write(text)
                        
                        print(f"\n💾 Result saved to: {output_file}")
                        return True
                    else:
                        print(f"❌ Result retrieval failed: {response.status_code}")
                        return False
                
                elif status == 'failed':
                    print(f"❌ Transcription failed: {status_data.get('error_message', 'Unknown error')}")
                    return False
                
                time.sleep(check_interval)
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return False
    
    print(f"❌ Transcription timed out after {max_wait_time} seconds")
    return False

if __name__ == "__main__":
    success = test_complete_transcription_flow()
    if success:
        print("\n🎉 SUCCESS: Complete transcription flow works!")
    else:
        print("\n💥 FAILED: Transcription flow has issues")
        sys.exit(1) 