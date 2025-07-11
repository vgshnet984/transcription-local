#!/usr/bin/env python3
"""
Combined test script for both Basic UI and Scripflow
Tests both UIs sequentially
"""

import subprocess
import time
import sys
import os

def test_both_uis():
    """Test both UIs."""
    print("🧪 Testing Both UIs")
    print("=" * 60)
    
    # Test Basic UI
    print("\n1️⃣ Testing Basic UI (port 8000)")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "test_basic_ui.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Basic UI test passed")
            print(result.stdout)
        else:
            print("❌ Basic UI test failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Basic UI test timed out")
        return False
    except Exception as e:
        print(f"❌ Basic UI test error: {e}")
        return False
    
    # Test Scripflow
    print("\n2️⃣ Testing Scripflow (port 8001)")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "test_scripflow.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Scripflow test passed")
            print(result.stdout)
        else:
            print("❌ Scripflow test failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Scripflow test timed out")
        return False
    except Exception as e:
        print(f"❌ Scripflow test error: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_both_uis()
    exit(0 if success else 1) 