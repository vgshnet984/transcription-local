#!/usr/bin/env python3
"""
Simple test to verify both UIs work and test with sample audio.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def test_basic_ui():
    """Test Basic UI startup."""
    print("🔍 Testing Basic UI...")
    print("=" * 40)
    
    try:
        # Start Basic UI in background
        process = subprocess.Popen([
            sys.executable, "start_basic_ui.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(10)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Basic UI is running")
            print(f"   Response: {response.json()}")
            return True, process
        else:
            print(f"❌ Basic UI health check failed: {response.status_code}")
            return False, process
            
    except Exception as e:
        print(f"❌ Basic UI test failed: {e}")
        return False, None

def test_scripflow_ui():
    """Test Scripflow UI startup."""
    print("\n🔍 Testing Scripflow UI...")
    print("=" * 40)
    
    try:
        # Start Scripflow UI in background
        process = subprocess.Popen([
            sys.executable, "start_scripflow.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(10)
        
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Scripflow UI is running")
            print(f"   Response: {response.json()}")
            return True, process
        else:
            print(f"❌ Scripflow UI health check failed: {response.status_code}")
            return False, process
            
    except Exception as e:
        print(f"❌ Scripflow UI test failed: {e}")
        return False, None

def test_sample_audio():
    """Test with sample audio file."""
    print("\n🔍 Testing with sample audio...")
    print("=" * 40)
    
    # Look for sample audio files
    sample_files = [
        "examples/sample_audio/test.wav",
        "uploads/test.wav",
        "test_audio.wav"
    ]
    
    test_file = None
    for file_path in sample_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("ℹ️  No sample audio file found")
        print("   Create a test file in uploads/ or examples/sample_audio/")
        return False
    
    print(f"✅ Found test file: {test_file}")
    
    # Test upload to Basic UI
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (os.path.basename(test_file), f, 'audio/wav')}
            data = {'language': 'en', 'transcription_engine': 'whisper'}
            
            response = requests.post(
                "http://localhost:8000/api/upload",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Basic UI upload successful")
                print(f"   File ID: {result.get('id')}")
                return True
            else:
                print(f"❌ Basic UI upload failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Basic UI upload test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Both UIs")
    print("=" * 60)
    
    basic_ok, basic_process = test_basic_ui()
    scripflow_ok, scripflow_process = test_scripflow_ui()
    
    if basic_ok and scripflow_ok:
        print("\n✅ Both UIs are running!")
        print("   Basic UI: http://localhost:8000")
        print("   Scripflow UI: http://localhost:8001")
        
        # Test with sample audio
        audio_ok = test_sample_audio()
        
        print("\n📊 Test Results:")
        print(f"   Basic UI: {'✅ PASS' if basic_ok else '❌ FAIL'}")
        print(f"   Scripflow UI: {'✅ PASS' if scripflow_ok else '❌ FAIL'}")
        print(f"   Sample Audio: {'✅ PASS' if audio_ok else '❌ FAIL'}")
        
        if basic_ok and scripflow_ok and audio_ok:
            print("\n🎉 All tests passed! Both UIs are working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n❌ UI startup tests failed")
        print(f"   Basic UI: {'✅ PASS' if basic_ok else '❌ FAIL'}")
        print(f"   Scripflow UI: {'✅ PASS' if scripflow_ok else '❌ FAIL'}")
    
    # Cleanup
    if basic_process:
        basic_process.terminate()
    if scripflow_process:
        scripflow_process.terminate()
    
    return basic_ok and scripflow_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 