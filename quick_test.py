#!/usr/bin/env python3
"""
Quick Test script for core transcription functionality
Tests CLI first, then optionally tests UIs
"""

import subprocess
import sys
import os
import datetime

def log_message(message, level="INFO"):
    """Log message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def quick_test():
    """Quick test of core functionality."""
    log_message("=" * 60, "TEST")
    log_message("Quick Test - Core Transcription Functionality", "TEST")
    log_message("=" * 60, "TEST")
    
    # Test CLI first
    log_message("1. Testing CLI Transcription", "TEST")
    log_message("-" * 40, "TEST")
    
    try:
        log_message("Running CLI test...", "RUN")
        result = subprocess.run([
            sys.executable, "test_transcription_cli.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            log_message("CLI test passed", "SUCCESS")
            print(result.stdout)
        else:
            log_message("CLI test failed", "ERROR")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        log_message("CLI test timed out", "ERROR")
        return False
    except Exception as e:
        log_message(f"CLI test error: {e}", "ERROR")
        return False
    
    log_message("=" * 60, "TEST")
    log_message("Quick test completed successfully!", "SUCCESS")
    log_message("=" * 60, "TEST")
    
    # Ask if user wants to test UIs
    print("\n" + "=" * 60)
    print("CLI test completed successfully!")
    print("=" * 60)
    print("\nDo you want to test the UIs? (y/n): ", end="")
    
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            log_message("Starting UI tests...", "TEST")
            return test_uis()
        else:
            log_message("Skipping UI tests", "INFO")
            return True
    except KeyboardInterrupt:
        log_message("Test interrupted by user", "INFO")
        return True

def test_uis():
    """Test both UIs."""
    log_message("2. Testing UIs", "TEST")
    log_message("-" * 40, "TEST")
    
    try:
        log_message("Running UI tests...", "RUN")
        result = subprocess.run([
            sys.executable, "test_both_uis_enhanced.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            log_message("UI tests passed", "SUCCESS")
            print(result.stdout)
            return True
        else:
            log_message("UI tests failed", "ERROR")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        log_message("UI tests timed out", "ERROR")
        return False
    except Exception as e:
        log_message(f"UI tests error: {e}", "ERROR")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1) 