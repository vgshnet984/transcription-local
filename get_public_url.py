#!/usr/bin/env python3
"""
Simple script to get public URL for Colab server.
"""

import time
import requests
import os

def get_public_url():
    """Get the public URL for the running server."""
    print("🌐 Getting Public URL for your Transcription Platform")
    print("=" * 60)
    
    # Wait a moment for server to fully start
    time.sleep(2)
    
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running successfully!")
            
            # Get Colab session info
            colab_session = os.environ.get('COLAB_SESSION_ID', 'unknown')
            
            print("\n🌐 Your URLs:")
            print("1. Internal URL: http://127.0.0.1:8000")
            print("2. Colab URL: https://[session-id]-8000.colab.research.google.com")
            
            print("\n📱 To get the exact public URL:")
            print("   Step 1: Go to Runtime → Manage sessions")
            print("   Step 2: Look for 'Connect to localhost:8000'")
            print("   Step 3: Click it to get your public URL")
            
            print("\n🎯 Alternative method:")
            print("   - Look in your Colab notebook output for any URL information")
            print("   - Check if there's a 'Connect to localhost' button")
            
            print("\n💡 Your transcription platform is now accessible!")
            print("   Anyone with the public URL can use your transcription service.")
            
            return True
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not running: {e}")
        print("\n💡 Make sure to start the server first with:")
        print("   !python src/main.py")
        return False

if __name__ == "__main__":
    get_public_url() 