#!/usr/bin/env python3
"""
Script to set HuggingFace token for speaker diarization.
"""

import sys
from config import config

def main():
    if len(sys.argv) != 2:
        print("Usage: python set_token.py <your_huggingface_token>")
        print("Example: python set_token.py YOUR_HF_TOKEN_HERE")
        return 1
    
    token = sys.argv[1]
    if not token.startswith('hf_'):
        print("Error: Token should start with 'hf_'")
        return 1
    
    config.set_hf_token(token)
    print(f"Token set successfully!")
    print(f"Token will be loaded automatically when starting the servers.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 