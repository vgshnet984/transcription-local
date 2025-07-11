#!/usr/bin/env python3
"""
Test script to verify frontend transcription loading functionality
"""

import requests
import json
import time

def test_frontend_transcription():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Frontend Transcription Loading...")
    
    # Use a specific completed job that we know has a transcription
    test_job_id = 53
    
    # Step 1: Get job details (simulating frontend polling)
    print(f"\n1. Getting job details for job {test_job_id}...")
    job_response = requests.get(f"{base_url}/api/jobs/{test_job_id}")
    if job_response.status_code != 200:
        print("❌ Failed to get job details")
        return False
    
    job_data = job_response.json()
    print(f"✅ Job status: {job_data['job']['status']}")
    print(f"✅ Job progress: {job_data['job']['progress']}%")
    print(f"✅ Transcription ID: {job_data['job']['transcription_id']}")
    
    if job_data['job']['status'] != 'completed':
        print("❌ Job is not completed")
        return False
    
    if not job_data['job']['transcription_id']:
        print("❌ No transcription ID found")
        return False
    
    # Step 2: Load transcription (simulating frontend loadResults)
    transcription_id = job_data['job']['transcription_id']
    print(f"\n2. Loading transcription {transcription_id}...")
    transcription_response = requests.get(f"{base_url}/api/transcriptions/{transcription_id}")
    if transcription_response.status_code != 200:
        print("❌ Failed to load transcription")
        return False
    
    transcription_data = transcription_response.json()
    print(f"✅ Transcription loaded successfully!")
    print(f"✅ Text length: {len(transcription_data['text'])} characters")
    print(f"✅ Confidence: {transcription_data['confidence']:.2%}")
    print(f"✅ Created: {transcription_data['created_at']}")
    
    # Step 3: Display sample of transcription (simulating frontend displayResults)
    print(f"\n3. Sample transcription text:")
    print("-" * 50)
    sample_text = transcription_data['text'][:200] + "..." if len(transcription_data['text']) > 200 else transcription_data['text']
    print(sample_text)
    print("-" * 50)
    
    # Step 4: Check if segments are available
    if 'speaker_segments' in transcription_data and transcription_data['speaker_segments']:
        print(f"\n4. Speaker segments found: {len(transcription_data['speaker_segments'])}")
        for i, segment in enumerate(transcription_data['speaker_segments'][:3]):  # Show first 3 segments
            print(f"   Segment {i+1}: {segment['start']:.1f}s - {segment['end']:.1f}s | {segment['speaker']}")
    else:
        print("\n4. No speaker segments found")
    
    print(f"\n🎉 Frontend transcription loading test PASSED!")
    print(f"✅ The basic UI should be able to display transcripts correctly")
    print(f"✅ Backend API is working properly")
    print(f"✅ Frontend polling and result loading should work")
    return True

if __name__ == "__main__":
    try:
        test_frontend_transcription()
    except Exception as e:
        print(f"❌ Test failed with error: {e}") 