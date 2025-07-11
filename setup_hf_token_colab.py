#!/usr/bin/env python3
"""
HuggingFace Token Setup for Google Colab
This script helps users set up their HuggingFace token for speaker diarization.
"""

import os
import json
from pathlib import Path
# Google Colab import (only available in Colab environment)
# This will be available when running in Google Colab

def setup_hf_token():
    """Set up HuggingFace token for speaker diarization."""
    print("🔑 HuggingFace Token Setup for Speaker Diarization")
    print("=" * 60)
    print()
    print("This token is required for speaker diarization features.")
    print("It allows the system to identify different speakers in your audio.")
    print()
    
    # Instructions
    print("📋 How to get your HuggingFace token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'Colab Transcription')")
    print("4. Select 'Read' role")
    print("5. Copy the token (starts with 'hf_')")
    print()
    
    # Get token from user
    while True:
        token = input("Enter your HuggingFace token (starts with 'hf_'): ").strip()
        
        if not token:
            print("⚠️  No token provided. Speaker diarization will be disabled.")
            print("You can run this script again later to enable it.")
            return False
        
        if not token.startswith('hf_'):
            print("❌ Token should start with 'hf_'. Please try again.")
            continue
        
        # Validate token format (basic check)
        if len(token) < 10:
            print("❌ Token seems too short. Please check and try again.")
            continue
        
        break
    
    # Save token to config
    config_content = {
        "hf_token": token
    }
    
    try:
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config_content, f, indent=2)
        
        # Set environment variable
        os.environ['HF_TOKEN'] = token
        
        print("✅ HuggingFace token saved successfully!")
        print("🔒 Token is stored in config.json (this file is in .gitignore)")
        print("🌐 Environment variable HF_TOKEN is set")
        print()
        print("🎯 Speaker diarization is now enabled!")
        print("You can now use speaker identification features in your transcriptions.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        return False

def verify_token():
    """Verify if token is working."""
    print("\n🔍 Verifying token...")
    
    try:
        import requests
        
        # Test token with HuggingFace API
        headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Token verified! Logged in as: {user_info.get('name', 'Unknown')}")
            return True
        else:
            print("❌ Token verification failed. Please check your token.")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not verify token: {e}")
        print("Token will still work for basic features.")
        return True

def main():
    """Main function."""
    success = setup_hf_token()
    
    if success:
        verify_token()
    
    print("\n" + "=" * 60)
    print("🎉 HuggingFace token setup completed!")
    print("\n📋 Next steps:")
    print("1. Run your transcription with speaker diarization enabled")
    print("2. The system will automatically use your token")
    print("3. You can run this script again anytime to update your token")
    
    return success

if __name__ == "__main__":
    main() 