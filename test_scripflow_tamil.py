#!/usr/bin/env python3
"""
Test script for Scripflow Tamil transcription
"""

import requests
import json
import time
import os

def test_scripflow_tamil():
    """Test Scripflow with Tamil language and different model configurations."""
    
    base_url = "http://localhost:8010"
    
    print("🧪 Testing Scripflow Tamil Transcription")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Scripflow server is running")
        else:
            print("❌ Scripflow server is not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Scripflow server: {e}")
        return
    
    # Test 2: Get engine info
    try:
        response = requests.get(f"{base_url}/api/engine/info")
        if response.status_code == 200:
            engine_info = response.json()
            print("✅ Engine info retrieved:")
            print(f"   - Available engines: {engine_info.get('available_engines', [])}")
            print(f"   - Current engine: {engine_info.get('current_engine', 'unknown')}")
            print(f"   - Current model: {engine_info.get('current_model', 'unknown')}")
        else:
            print("❌ Failed to get engine info")
    except Exception as e:
        print(f"❌ Error getting engine info: {e}")
    
    # Test 3: List files
    try:
        response = requests.get(f"{base_url}/api/files")
        if response.status_code == 200:
            files = response.json()
            print(f"✅ Found {len(files)} uploaded files")
            if files:
                print("   Available files:")
                for file in files[:3]:  # Show first 3 files
                    print(f"   - {file['filename']} (ID: {file['id']})")
        else:
            print("❌ Failed to list files")
    except Exception as e:
        print(f"❌ Error listing files: {e}")
    
    # Test 4: Test different model configurations for Tamil
    print("\n🔧 Testing Tamil Language Configurations:")
    print("-" * 40)
    
    # Configuration 1: Fast setup (recommended for Tamil)
    config_fast = {
        "language": "tamil",
        "transcription_engine": "faster-whisper",
        "whisper_model": "base",
        "device": "cpu",
        "enable_speaker_diarization": False,
        "show_romanized_text": False
    }
    
    # Configuration 2: Standard Whisper (slower)
    config_standard = {
        "language": "tamil", 
        "transcription_engine": "whisper",
        "whisper_model": "base",
        "device": "cpu",
        "enable_speaker_diarization": False,
        "show_romanized_text": False
    }
    
    # Configuration 3: Large model (very slow)
    config_large = {
        "language": "tamil",
        "transcription_engine": "whisper", 
        "whisper_model": "large",
        "device": "cpu",
        "enable_speaker_diarization": False,
        "show_romanized_text": False
    }
    
    configs = [
        ("Fast (faster-whisper + base)", config_fast),
        ("Standard (whisper + base)", config_standard), 
        ("Large (whisper + large)", config_large)
    ]
    
    for config_name, config in configs:
        print(f"\n📋 Testing: {config_name}")
        print(f"   Engine: {config['transcription_engine']}")
        print(f"   Model: {config['whisper_model']}")
        print(f"   Language: {config['language']}")
        
        # Note: This would require an actual audio file to test
        print("   ⚠️  Note: Upload a Tamil audio file to test this configuration")
        print("   💡 Recommendation: Use 'Fast' configuration for best performance")
    
    print("\n🎯 Recommendations for Tamil Transcription:")
    print("-" * 40)
    print("1. Use 'faster-whisper' engine for faster processing")
    print("2. Start with 'base' model for quick results")
    print("3. Use 'large' model only if you need maximum accuracy")
    print("4. Consider using 'medium' model as a good balance")
    print("5. Enable speaker diarization only if needed")
    
    print("\n🌐 Scripflow UI Access:")
    print("-" * 40)
    print(f"   URL: {base_url}")
    print("   Features:")
    print("   - Advanced UI with Windows 11 styling")
    print("   - Multiple transcription engines")
    print("   - Speaker diarization support")
    print("   - Real-time progress tracking")
    print("   - Export options (TXT, SRT, VTT)")

if __name__ == "__main__":
    test_scripflow_tamil() 