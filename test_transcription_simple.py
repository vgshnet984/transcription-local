import requests
import json
import time
import os

def test_simple_transcription():
    """Test transcription with a smaller, faster model."""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Simple Transcription (Base Model)")
    print("=" * 50)
    
    # Test 1: Check server status
    print("\n1. Checking server status...")
    try:
        response = requests.get(f"{base_url}/engine-status", timeout=10)
        if response.status_code == 200:
            print("✅ Server is running")
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
            print(f"✅ File uploaded successfully (ID: {file_id})")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    # Test 3: Start transcription with base model (faster)
    print("\n3. Starting transcription with base model...")
    transcribe_data = {
        'file_id': file_id,
        'engine': 'faster-whisper',
        'model_size': 'base',  # Use base instead of large-v3
        'language': 'en',
        'vad_method': 'simple',
        'enable_speaker_diarization': False,  # Disable for speed
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
            print(f"✅ Transcription started (Job ID: {job_id})")
        else:
            print(f"❌ Transcription start failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Transcription start error: {e}")
        return False
    
    # Test 4: Monitor with shorter timeout
    print("\n4. Monitoring transcription (max 2 minutes)...")
    max_wait_time = 120  # 2 minutes max
    check_interval = 3   # Check every 3 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{base_url}/api/jobs/{job_id}/status", timeout=5)
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
                    
                    # Get result
                    response = requests.get(f"{base_url}/api/jobs/{job_id}/result", timeout=10)
                    if response.status_code == 200:
                        result_data = response.json()
                        text = result_data.get('text', '')
                        print(f"✅ Result retrieved!")
                        print(f"   Text length: {len(text)} characters")
                        print(f"\n📝 Preview: {text[:200]}...")
                        
                        # Save result
                        output_file = f"transcript_output/simple_transcript_{job_id}.txt"
                        os.makedirs("transcript_output", exist_ok=True)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                        print(f"\n💾 Saved to: {output_file}")
                        return True
                    else:
                        print(f"❌ Result retrieval failed")
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
    success = test_simple_transcription()
    if success:
        print("\n🎉 SUCCESS: Simple transcription works!")
    else:
        print("\n💥 FAILED: Simple transcription has issues") 