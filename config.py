#!/usr/bin/env python3
"""
Configuration management for the transcription platform.
Handles HuggingFace token, environment variables, and settings.
"""

import os
import json
from pathlib import Path
from typing import Optional

class Config:
    """Configuration manager for the transcription platform."""
    
    def __init__(self):
        self.config_file = Path("config.json")
        self.hf_token: Optional[str] = None
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or environment."""
        # Try to load from config file first
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self.hf_token = config_data.get('hf_token')
                    print(f"Loaded HuggingFace token from config file")
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # Fallback to environment variable
        if not self.hf_token:
            self.hf_token = os.environ.get('HF_TOKEN')
            if self.hf_token:
                print(f"Loaded HuggingFace token from environment variable")
        
        # Set environment variable for pyannote.audio
        if self.hf_token:
            os.environ['HF_TOKEN'] = self.hf_token
            print(f"HuggingFace token set in environment")
        else:
            print("Warning: No HuggingFace token found. Speaker diarization may not work.")
    
    def save_config(self):
        """Save configuration to file."""
        config_data = {
            'hf_token': self.hf_token
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def set_hf_token(self, token: str):
        """Set the HuggingFace token."""
        self.hf_token = token
        os.environ['HF_TOKEN'] = token
        self.save_config()
        print(f"HuggingFace token set and saved")

# Global config instance
config = Config()

# Export for easy access
HF_TOKEN = config.hf_token 