# CUDA and cuDNN Setup Guide for Transcription Platform

## Current System Configuration

Based on the analysis of the working system:
- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **CUDA Version**: 12.8
- **Driver Version**: 572.16
- **cuDNN Version**: 8.x (based on DLL files)

## Prerequisites

### 1. System Requirements
- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB free space for CUDA toolkit

### 2. Check GPU Compatibility
```bash
# Check if you have an NVIDIA GPU
nvidia-smi
```
If this command fails, you don't have an NVIDIA GPU or drivers installed.

## Step-by-Step Installation

### Step 1: Install NVIDIA GPU Drivers

1. **Download NVIDIA Drivers**
   - Go to: https://www.nvidia.com/Download/index.aspx
   - Select your GPU model and OS
   - Download and install the latest driver

2. **Verify Installation**
   ```bash
   nvidia-smi
   ```
   Should show your GPU and driver version.

### Step 2: Install CUDA Toolkit

1. **Download CUDA Toolkit 12.8**
   - Go to: https://developer.nvidia.com/cuda-downloads
   - Select:
     - Operating System: Windows
     - Architecture: x86_64
     - Version: 11 or 10 (Windows)
     - Installer Type: exe (local)
   - Download the installer

2. **Install CUDA Toolkit**
   ```bash
   # Run the downloaded installer
   # Default installation path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\
   ```

3. **Add CUDA to PATH**
   - Open System Properties → Environment Variables
   - Add to PATH:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
     ```

4. **Verify CUDA Installation**
   ```bash
   nvcc --version
   ```
   Should show CUDA version 12.8.

### Step 3: Install cuDNN

1. **Download cuDNN 8.x**
   - Go to: https://developer.nvidia.com/cudnn
   - **Note**: Requires free NVIDIA developer account
   - Download cuDNN v8.x for CUDA 12.x

2. **Extract and Install cuDNN**
   ```bash
   # Extract the downloaded zip file
   # Copy files to CUDA installation directory:
   
   # Copy from extracted folder to CUDA directory:
   copy cuda\bin\* "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\"
   copy cuda\include\* "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\"
   copy cuda\lib\x64\* "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64\"
   ```

3. **Alternative: Install to Custom Directory**
   ```bash
   # Create custom cuDNN directory
   mkdir C:\cudnn
   mkdir C:\cudnn\bin
   mkdir C:\cudnn\include
   mkdir C:\cudnn\lib
   
   # Copy files
   copy cuda\bin\* C:\cudnn\bin\
   copy cuda\include\* C:\cudnn\include\
   copy cuda\lib\x64\* C:\cudnn\lib\
   
   # Add to PATH
   # Add C:\cudnn\bin to your system PATH
   ```

### Step 4: Verify Installation

1. **Check CUDA Installation**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Check cuDNN Installation**
   ```bash
   # Verify cuDNN DLLs are accessible
   dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudnn*.dll"
   # or if using custom directory:
   dir C:\cudnn\bin\cudnn*.dll
   ```

3. **Test with Python**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"cuDNN version: {torch.backends.cudnn.version()}")
   ```

## Environment Variables

Add these to your system environment variables:

```bash
# System Environment Variables
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# Add to PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
```

## PyTorch Installation with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8 (if 12.1 doesn't work)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Common Issues

1. **"Could not locate cudnn_cnn_infer64_8.dll"**
   - Ensure cuDNN is properly installed
   - Check PATH includes cuDNN bin directory
   - Verify DLL files exist in the expected location

2. **CUDA not detected by PyTorch**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.version.cuda)         # Should show CUDA version
   ```

3. **Driver version mismatch**
   - Update NVIDIA drivers to latest version
   - Ensure driver version supports your CUDA version

4. **Memory issues**
   - Reduce batch size in transcription settings
   - Use smaller models (tiny, base instead of large)
   - Monitor GPU memory usage with `nvidia-smi`

### Verification Commands

```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Check cuDNN
python -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version()}')"
```

## Performance Optimization

### GPU Memory Management
```python
# In your transcription code
import torch

# Set memory fraction (use 80% of GPU memory)
torch.cuda.set_per_process_memory_fraction(0.8)

# Clear cache if needed
torch.cuda.empty_cache()
```

### Model Selection for Different GPUs
- **RTX 3060 (12GB)**: Can use large models
- **GTX 1660 (6GB)**: Use medium or smaller models
- **GTX 1060 (3GB)**: Use tiny or base models only

## Alternative: CPU-Only Setup

If you don't have an NVIDIA GPU:

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Transcription will work but be slower
# Use smaller models for better performance
```

## Files to Check After Installation

After successful installation, verify these files exist:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\
├── nvcc.exe
├── cudnn_adv_infer64_8.dll
├── cudnn_adv_train64_8.dll
├── cudnn_cnn_infer64_8.dll
└── cudnn_cnn_train64_8.dll

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\
└── cudnn.h

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64\
└── cudnn.lib
```

## Next Steps

After completing CUDA/cuDNN setup:

1. **Install project dependencies**:
   ```bash
   pip install -r requirements_local.txt
   ```

2. **Test transcription with GPU**:
   ```bash
   python test_transcription_simple.py
   ```

3. **Verify GPU acceleration**:
   - Check that transcription is faster than CPU
   - Monitor GPU usage with `nvidia-smi`

## Support

If you encounter issues:
1. Check NVIDIA forums: https://forums.developer.nvidia.com/
2. Verify your GPU supports CUDA: https://developer.nvidia.com/cuda-gpus
3. Check PyTorch documentation: https://pytorch.org/get-started/locally/ 