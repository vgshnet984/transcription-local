#!/usr/bin/env python3
"""
Enhanced Combined Test script for both Basic UI and Scripflow
Tests both UIs sequentially with detailed logging
"""

import subprocess
import time
import sys
import os
import datetime

def log_message(message, level="INFO"):
    """Log message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def test_both_uis_enhanced():
    """Test both UIs with enhanced logging."""
    log_message("=" * 80, "TEST")
    log_message("Testing Both UIs - Enhanced Version", "TEST")
    log_message("=" * 80, "TEST")
    
    # Test Basic UI
    log_message("1. Testing Basic UI (port 8000)", "TEST")
    log_message("-" * 50, "TEST")
    
    try:
        log_message("Running Basic UI test...", "RUN")
        result = subprocess.run([
            sys.executable, "test_basic_ui_enhanced.py"
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            log_message("Basic UI test passed", "SUCCESS")
            print(result.stdout)
        else:
            log_message("Basic UI test failed", "ERROR")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        log_message("Basic UI test timed out", "ERROR")
        return False
    except Exception as e:
        log_message(f"Basic UI test error: {e}", "ERROR")
        return False
    
    # Test Scripflow
    log_message("2. Testing Scripflow (port 8001)", "TEST")
    log_message("-" * 50, "TEST")
    
    try:
        log_message("Running Scripflow test...", "RUN")
        result = subprocess.run([
            sys.executable, "test_scripflow_enhanced.py"
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            log_message("Scripflow test passed", "SUCCESS")
            print(result.stdout)
        else:
            log_message("Scripflow test failed", "ERROR")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        log_message("Scripflow test timed out", "ERROR")
        return False
    except Exception as e:
        log_message(f"Scripflow test error: {e}", "ERROR")
        return False
    
    log_message("=" * 80, "TEST")
    log_message("All tests completed successfully!", "SUCCESS")
    log_message("=" * 80, "TEST")
    return True

if __name__ == "__main__":
    success = test_both_uis_enhanced()
    exit(0 if success else 1) 