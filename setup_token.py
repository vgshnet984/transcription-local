#!/usr/bin/env python3
"""
Interactive script to set up HuggingFace token for new users.
"""

import os
import sys
from pathlib import Path

def main():
    print("🤗 HuggingFace Token Setup")
    print("=" * 40)
    print()
    print("This script will help you set up your HuggingFace token for speaker diarization.")
    print("You can get a free token from: https://huggingface.co/settings/tokens")
    print()
    
    # Check if config.json already exists
    config_file = Path("config.json")
    if config_file.exists():
        print("⚠️  config.json already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return 0
    
    # Get token from user
    while True:
        token = input("Enter your HuggingFace token (starts with 'hf_'): ").strip()
        
        if not token:
            print("❌ Token cannot be empty. Please try again.")
            continue
            
        if not token.startswith('hf_'):
            print("❌ Token should start with 'hf_'. Please try again.")
            continue
            
        break
    
    # Create config.json
    config_content = f'''{{
  "hf_token": "{token}"
}}'''
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ Token saved to {config_file}")
        print("🔒 This file is already in .gitignore to keep your token secure.")
        print()
        print("You can now run the transcription platform!")
        return 0
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 