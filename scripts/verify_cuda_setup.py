#!/usr/bin/env python3
"""
CUDA and cuDNN Setup Verification Script
Run this script to verify your CUDA/cuDNN installation is working correctly.
"""

import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_nvidia_driver():
    """Check if NVIDIA drivers are installed"""
    print("🔍 Checking NVIDIA Driver...")
    success, output, error = run_command("nvidia-smi")
    
    if success:
        print("✅ NVIDIA Driver: OK")
        print(f"   Output: {output.split('Driver Version:')[1].split()[0] if 'Driver Version:' in output else 'Unknown'}")
        return True
    else:
        print("❌ NVIDIA Driver: NOT FOUND")
        print(f"   Error: {error}")
        return False

def check_cuda_toolkit():
    """Check if CUDA toolkit is installed"""
    print("\n🔍 Checking CUDA Toolkit...")
    success, output, error = run_command("nvcc --version")
    
    if success:
        print("✅ CUDA Toolkit: OK")
        print(f"   Version: {output.split('release ')[1].split(',')[0] if 'release ' in output else 'Unknown'}")
        return True
    else:
        print("❌ CUDA Toolkit: NOT FOUND")
        print(f"   Error: {error}")
        return False

def check_cudnn_files():
    """Check if cuDNN files are present"""
    print("\n🔍 Checking cuDNN Files...")
    
    # Check standard CUDA installation path
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\cudnn\bin"
    ]
    
    cudnn_files = [
        "cudnn_adv_infer64_8.dll",
        "cudnn_cnn_infer64_8.dll"
    ]
    
    found_path = None
    for path in cuda_paths:
        if os.path.exists(path):
            all_files_present = True
            for file in cudnn_files:
                if not os.path.exists(os.path.join(path, file)):
                    all_files_present = False
                    break
            
            if all_files_present:
                found_path = path
                break
    
    if found_path:
        print("✅ cuDNN Files: OK")
        print(f"   Location: {found_path}")
        return True
    else:
        print("❌ cuDNN Files: NOT FOUND")
        print("   Checked paths:")
        for path in cuda_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"   {exists} {path}")
        return False

def check_pytorch_cuda():
    """Check if PyTorch can use CUDA"""
    print("\n🔍 Checking PyTorch CUDA Support...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        try:
            cuda_version = torch.version.cuda
        except AttributeError:
            cuda_version = "Unknown"
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"
        
        if cuda_available:
            print("✅ PyTorch CUDA: OK")
            print(f"   CUDA Available: {cuda_available}")
            print(f"   CUDA Version: {cuda_version}")
            print(f"   cuDNN Version: {cudnn_version}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            
            if torch.cuda.device_count() > 0:
                print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            return True
        else:
            print("❌ PyTorch CUDA: NOT AVAILABLE")
            print(f"   CUDA Version: {cuda_version}")
            return False
            
    except ImportError:
        print("❌ PyTorch: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"❌ PyTorch CUDA Check Failed: {e}")
        return False

def check_environment_variables():
    """Check CUDA environment variables"""
    print("\n🔍 Checking Environment Variables...")
    
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_path_v12_8 = os.environ.get('CUDA_PATH_V12_8')
    path = os.environ.get('PATH', '')
    
    if cuda_path:
        print(f"✅ CUDA_PATH: {cuda_path}")
    else:
        print("❌ CUDA_PATH: NOT SET")
    
    if cuda_path_v12_8:
        print(f"✅ CUDA_PATH_V12_8: {cuda_path_v12_8}")
    else:
        print("❌ CUDA_PATH_V12_8: NOT SET")
    
    # Check if CUDA bin is in PATH
    cuda_in_path = any('cuda' in p.lower() and 'bin' in p.lower() for p in path.split(os.pathsep))
    if cuda_in_path:
        print("✅ CUDA in PATH: OK")
    else:
        print("❌ CUDA in PATH: NOT FOUND")
    
    return bool(cuda_path or cuda_path_v12_8 or cuda_in_path)

def test_transcription_gpu():
    """Test if transcription works with GPU"""
    print("\n🔍 Testing Transcription with GPU...")
    
    try:
        # Simple test to see if GPU transcription works
        import torch
        from faster_whisper import WhisperModel
        
        if not torch.cuda.is_available():
            print("❌ GPU not available for testing")
            return False
        
        # Try to load a small model on GPU
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        print("✅ GPU Transcription: OK")
        print("   Successfully loaded Whisper model on GPU")
        return True
        
    except ImportError:
        print("❌ faster-whisper: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"❌ GPU Transcription Test Failed: {e}")
        return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("CUDA and cuDNN Setup Verification")
    print("=" * 60)
    
    checks = [
        ("NVIDIA Driver", check_nvidia_driver),
        ("CUDA Toolkit", check_cuda_toolkit),
        ("cuDNN Files", check_cudnn_files),
        ("Environment Variables", check_environment_variables),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("GPU Transcription", test_transcription_gpu)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your CUDA/cuDNN setup is working correctly.")
        print("You can now run the transcription platform with GPU acceleration.")
    else:
        print("⚠️  Some checks failed. Please review the setup guide:")
        print("   https://github.com/vgshnet984/transcription-local/blob/main/CUDA_CUDNN_SETUP.md")
    
    print("\nFor detailed setup instructions, see: CUDA_CUDNN_SETUP.md")

if __name__ == "__main__":
    main() 