#!/usr/bin/env python3
"""
cuDNN Download and Setup Script
Automatically downloads and sets up cuDNN files for GPU acceleration.
"""

import os
import sys
import zipfile
import requests
import shutil
from pathlib import Path

def check_cudnn_installation():
    """Check if cuDNN is already installed"""
    cudnn_paths = [
        r"C:\cudnn\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    ]
    
    required_files = [
        "cudnn_adv_infer64_8.dll",
        "cudnn_cnn_infer64_8.dll"
    ]
    
    for path in cudnn_paths:
        if os.path.exists(path):
            all_files_present = True
            for file in required_files:
                if not os.path.exists(os.path.join(path, file)):
                    all_files_present = False
                    break
            
            if all_files_present:
                print(f"✅ cuDNN already installed at: {path}")
                return True, path
    
    return False, None

def download_cudnn():
    """Download cuDNN from NVIDIA (requires manual download)"""
    print("📥 cuDNN Download Instructions:")
    print("=" * 50)
    print("1. Go to: https://developer.nvidia.com/cudnn")
    print("2. Sign up/login to NVIDIA Developer Program (free)")
    print("3. Download cuDNN for your CUDA version:")
    print("   - For CUDA 12.x: Download cuDNN v8.x")
    print("   - For CUDA 11.x: Download cuDNN v8.x")
    print("4. Save the zip file to: C:\\cudnn_download.zip")
    print("5. Run this script again to extract and install")
    print("=" * 50)
    
    # Check if download file exists
    download_path = r"C:\cudnn_download.zip"
    if os.path.exists(download_path):
        print(f"✅ Found download file: {download_path}")
        return extract_cudnn(download_path)
    else:
        print(f"❌ Download file not found: {download_path}")
        print("Please download cuDNN manually and save as: C:\\cudnn_download.zip")
        return False

def extract_cudnn(zip_path):
    """Extract cuDNN files to C:\cudnn"""
    print(f"📦 Extracting cuDNN from: {zip_path}")
    
    try:
        # Create C:\cudnn directory
        cudnn_dir = r"C:\cudnn"
        os.makedirs(cudnn_dir, exist_ok=True)
        os.makedirs(os.path.join(cudnn_dir, "bin"), exist_ok=True)
        os.makedirs(os.path.join(cudnn_dir, "include"), exist_ok=True)
        os.makedirs(os.path.join(cudnn_dir, "lib"), exist_ok=True)
        
        # Extract files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(r"C:\cudnn_temp")
        
        # Find and copy files
        temp_dir = r"C:\cudnn_temp"
        
        # Look for cuda folder structure
        cuda_folders = []
        for root, dirs, files in os.walk(temp_dir):
            if "cuda" in root.lower():
                cuda_folders.append(root)
        
        if not cuda_folders:
            print("❌ Could not find cuda folder structure in zip file")
            return False
        
        cuda_root = cuda_folders[0]
        print(f"📁 Found CUDA structure at: {cuda_root}")
        
        # Copy files
        copy_operations = [
            (os.path.join(cuda_root, "bin"), os.path.join(cudnn_dir, "bin")),
            (os.path.join(cuda_root, "include"), os.path.join(cudnn_dir, "include")),
            (os.path.join(cuda_root, "lib", "x64"), os.path.join(cudnn_dir, "lib"))
        ]
        
        for src, dst in copy_operations:
            if os.path.exists(src):
                print(f"📋 Copying {src} to {dst}")
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                print(f"⚠️  Source not found: {src}")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print("✅ cuDNN extraction completed!")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def add_to_path():
    """Add C:\cudnn\bin to PATH environment variable"""
    print("🔧 Adding C:\\cudnn\\bin to PATH...")
    
    try:
        import winreg
        
        # Get current PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE)
        path_value, _ = winreg.QueryValueEx(key, "Path")
        
        cudnn_path = r"C:\cudnn\bin"
        
        if cudnn_path not in path_value:
            new_path = path_value + ";" + cudnn_path
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            print("✅ Added C:\\cudnn\\bin to system PATH")
            print("⚠️  You may need to restart your terminal for changes to take effect")
        else:
            print("✅ C:\\cudnn\\bin already in PATH")
        
        winreg.CloseKey(key)
        return True
        
    except Exception as e:
        print(f"❌ Failed to update PATH: {e}")
        print("⚠️  Please manually add C:\\cudnn\\bin to your system PATH")
        return False

def verify_installation():
    """Verify cuDNN installation"""
    print("\n🔍 Verifying cuDNN installation...")
    
    cudnn_bin = r"C:\cudnn\bin"
    required_files = [
        "cudnn_adv_infer64_8.dll",
        "cudnn_cnn_infer64_8.dll"
    ]
    
    if not os.path.exists(cudnn_bin):
        print("❌ C:\\cudnn\\bin directory not found")
        return False
    
    all_files_present = True
    for file in required_files:
        file_path = os.path.join(cudnn_bin, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_files_present = False
    
    return all_files_present

def main():
    """Main function"""
    print("=" * 60)
    print("cuDNN Download and Setup Script")
    print("=" * 60)
    
    # Check if already installed
    installed, path = check_cudnn_installation()
    if installed:
        print(f"✅ cuDNN is already properly installed at: {path}")
        return True
    
    print("❌ cuDNN not found. Starting download process...")
    
    # Check for existing download
    download_path = r"C:\cudnn_download.zip"
    if os.path.exists(download_path):
        print("📦 Found existing download file")
        if extract_cudnn(download_path):
            add_to_path()
            if verify_installation():
                print("\n🎉 cuDNN installation completed successfully!")
                print("You can now run the transcription platform with GPU acceleration.")
                return True
    else:
        download_cudnn()
    
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Setup incomplete. Please follow the manual instructions above.")
    input("\nPress Enter to exit...") 