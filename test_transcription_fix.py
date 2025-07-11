import requests
import json
import time
import os

def test_transcription_flow():
    base_url = "http://localhost:8000"
    
    # Test 1: Check engine status
    print("1. Testing engine status...")
    response = requests.get(f"{base_url}/engine-status")
    if response.status_code == 200:
        print("✅ Engine status OK")
    else:
        print(f"❌ Engine status failed: {response.status_code}")
        return
    
    # Test 2: Upload a test file (if available)
    test_file = "examples/sample_audio/Hello.m4a"  # Use available file
    if not os.path.exists(test_file):
        print(f"⚠️ Test file not found: {test_file}")
        print("Skipping upload test - you can test manually via the UI")
        return
    
    print("2. Testing file upload...")
    with open(test_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/api/upload", files=files)
    
    if response.status_code == 200:
        upload_data = response.json()
        file_id = upload_data['id']
        print(f"✅ File uploaded successfully, ID: {file_id}")
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        return
    
    # Test 3: Start transcription
    print("3. Testing transcription start...")
    transcribe_data = {
        'file_id': file_id,
        'engine': 'faster-whisper',
        'model_size': 'base',
        'language': 'en',
        'vad_method': 'simple',
        'enable_speaker_diarization': False,
        'no_filtering': False,
        'suppress_logs': True
    }
    
    response = requests.post(f"{base_url}/api/transcribe", 
                           json=transcribe_data,
                           headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ Transcription started successfully, Job ID: {job_id}")
        
        # Test 4: Check job status
        print("4. Testing job status...")
        for i in range(10):  # Check status for up to 10 seconds
            time.sleep(1)
            response = requests.get(f"{base_url}/api/jobs/{job_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                print(f"   Status: {status_data['status']}, Progress: {status_data['progress']}%")
                
                if status_data['status'] in ['completed', 'failed']:
                    if status_data['status'] == 'completed':
                        print("✅ Transcription completed successfully!")
                        
                        # Test 5: Get result
                        print("5. Testing result retrieval...")
                        response = requests.get(f"{base_url}/api/jobs/{job_id}/result")
                        if response.status_code == 200:
                            result_data = response.json()
                            print(f"✅ Result retrieved: {len(result_data['text'])} characters")
                            print(f"   Text preview: {result_data['text'][:100]}...")
                        else:
                            print(f"❌ Result retrieval failed: {response.status_code}")
                    else:
                        print(f"❌ Transcription failed: {status_data.get('error_message', 'Unknown error')}")
                    break
            else:
                print(f"❌ Status check failed: {response.status_code}")
                break
    else:
        print(f"❌ Transcription start failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_transcription_flow() 