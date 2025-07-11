import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple
import librosa
from pydub import AudioSegment
import soundfile as sf
from loguru import logger
import tempfile
import subprocess
import numpy as np

from config import settings


class AudioProcessor:
    """Audio processing utilities for file validation and handling."""
    
    def __init__(self):
        self.supported_formats = settings.supported_formats_list
        self.max_file_size = settings.max_file_size_bytes
        self.sample_rate = settings.sample_rate
    
    def validate(self, file_path: str) -> Dict:
        """
        Validate audio file format, size, and basic properties.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results and file info
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File not found"}
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return {
                    "valid": False, 
                    "error": f"File too large: {file_size} bytes (max: {self.max_file_size})"
                }
            
            # Check file format
            file_ext = Path(file_path).suffix.lower().lstrip(".")
            if file_ext not in self.supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {file_ext}"
                }
            
            # Get audio properties
            meta = self.get_metadata(file_path)
            
            return {
                "valid": True,
                "file_size": file_size,
                "format": file_ext,
                "duration": meta.get("duration"),
                "sample_rate": meta.get("sample_rate"),
                "channels": meta.get("channels")
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_metadata(self, file_path: str) -> Dict:
        """
        Get audio file information using librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio properties
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Get duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get number of channels (librosa loads as mono, so we need to check original)
            audio_segment = AudioSegment.from_file(file_path)
            channels = audio_segment.channels
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": channels,
                "format": Path(file_path).suffix.lower().lstrip('.')
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}
    
    def save_uploaded_file(self, uploaded_file, original_filename: str) -> Tuple[str, Dict]:
        """
        Save uploaded file to local storage (FAST VERSION - no heavy processing).
        
        Args:
            uploaded_file: FastAPI UploadFile object or file-like object
            original_filename: Original filename
            
        Returns:
            Tuple of (saved_file_path, file_info)
        """
        try:
            # Generate unique filename
            file_ext = Path(original_filename).suffix.lower()
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            
            # Create upload directory if it doesn't exist
            upload_dir = Path(settings.upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = upload_dir / unique_filename
            
            with open(file_path, "wb") as buffer:
                # Handle both UploadFile objects and file-like objects
                if hasattr(uploaded_file, 'file'):
                    shutil.copyfileobj(uploaded_file.file, buffer)
                else:
                    shutil.copyfileobj(uploaded_file, buffer)
            
            # FAST VALIDATION - only check file size and format, skip audio processing
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                os.remove(file_path)
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
            file_ext_check = Path(file_path).suffix.lower().lstrip(".")
            if file_ext_check not in self.supported_formats:
                os.remove(file_path)
                raise ValueError(f"Unsupported format: {file_ext_check}")
            
            # Basic file info without heavy audio processing
            file_info = {
                "filename": unique_filename,
                "original_filename": original_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "format": file_ext_check,
                "duration": None,  # Will be extracted later during transcription
                "sample_rate": None,  # Will be extracted later during transcription
                "channels": None  # Will be extracted later during transcription
            }
            
            logger.info(f"File saved successfully (fast mode): {file_info}")
            return str(file_path), file_info
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise
    
    def preprocess_audio(self, file_path: str) -> str:
        """
        Preprocess audio file for better transcription quality.
        Includes denoising, normalization, and VAD if enabled.
        
        Args:
            file_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            if not settings.enable_audio_preprocessing:
                logger.info("Audio preprocessing disabled, returning original file")
                return file_path
            
            logger.info(f"Preprocessing audio file: {file_path}")
            
            # Create temporary file for preprocessing
            temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            
            # Build FFmpeg command for preprocessing
            ffmpeg_cmd = [
                'ffmpeg', '-i', file_path, '-y',  # -y to overwrite output
                '-ar', str(settings.target_sample_rate),  # Set sample rate
                '-ac', '1'  # Convert to mono
            ]
            
            # Add audio filters for preprocessing
            filters = []
            
            if settings.enable_denoising:
                # High-pass and low-pass filters for denoising
                filters.append(f"highpass=f={settings.highpass_freq}")
                filters.append(f"lowpass=f={settings.lowpass_freq}")
                logger.info(f"Added denoising filters: highpass={settings.highpass_freq}Hz, lowpass={settings.lowpass_freq}Hz")
            
            if settings.enable_normalization:
                # Normalize audio levels
                filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
                logger.info("Added audio normalization")
            
            # Apply filters if any
            if filters:
                ffmpeg_cmd.extend(['-af', ','.join(filters)])
            
            ffmpeg_cmd.append(temp_output)
            
            # Execute FFmpeg command
            logger.info(f"Running FFmpeg preprocessing: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg preprocessing failed: {result.stderr}")
                # Fallback to original file
                return file_path
            
            logger.info(f"Audio preprocessing completed: {temp_output}")
            return temp_output
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Fallback to original file
            return file_path
    
    def apply_vad(self, file_path: str) -> str:
        """
        Apply Voice Activity Detection to remove silence and non-speech segments.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Path to VAD-processed audio file
        """
        try:
            if not settings.enable_vad:
                logger.info("VAD disabled, returning original file")
                return file_path
            
            logger.info(f"Applying VAD to: {file_path}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=settings.target_sample_rate)
            
            # Apply VAD using librosa's voice activity detection
            # This is a simple approach - for more advanced VAD, consider using webrtcvad or similar
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate spectral centroid as a simple VAD indicator
            spectral_centroids = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length
            )[0]
            
            # Calculate energy
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Combine features for VAD
            vad_threshold = np.percentile(spectral_centroids, 20)  # Use 20th percentile as threshold
            energy_threshold = np.percentile(energy, 30)  # Use 30th percentile as threshold
            
            # Create VAD mask
            vad_mask = (spectral_centroids > vad_threshold) & (energy > energy_threshold)
            
            # Apply mask to audio
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            vad_frames = frames[:, vad_mask]
            
            if vad_frames.size == 0:
                logger.warning("VAD removed all audio, returning original file")
                return file_path
            
            # Reconstruct audio from VAD frames
            vad_audio = librosa.frames_to_samples(vad_frames, hop_length=hop_length)
            
            # Save VAD-processed audio
            temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            sf.write(temp_output, vad_audio, sr)
            
            logger.info(f"VAD processing completed: {temp_output}")
            return temp_output
            
        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            return file_path
    
    def convert_to_wav(self, file_path: str, target_sr=16000, mono=True) -> str:
        """
        Convert audio file to specified format.
        
        Args:
            input_path: Path to input audio file
            output_format: Desired output format
            
        Returns:
            Path to converted file
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(file_path)
            
            if mono:
                audio = audio.set_channels(1)
            
            audio = audio.set_frame_rate(target_sr)
            
            # Generate output path
            out_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            
            # Export to new format
            audio.export(out_path, format='wav')
            
            logger.info(f"Converted {file_path} to {out_path} ({target_sr}Hz, mono={mono})")
            return out_path
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise
    
    def normalize_audio(self, file_path: str, target_sr: Optional[int] = None) -> str:
        """
        Normalize audio to target sample rate.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (uses settings if None)
            
        Returns:
            Path to normalized file
        """
        try:
            target_sr = target_sr or self.sample_rate
            
            # Load audio
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # Normalize audio levels
            y_normalized = librosa.util.normalize(y)
            
            # Save normalized audio
            temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            sf.write(temp_output, y_normalized, target_sr)
            
            logger.info(f"Audio normalized: {temp_output}")
            return temp_output
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return file_path
    
    def cleanup_file(self, file_path: str):
        """Clean up temporary file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")

    def optimize_for_transcription(self, audio_path):
        """Reduce audio quality to speed up processing"""
        import librosa
        
        # Load and resample to 16kHz (Whisper's preferred rate)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Save optimized version
        optimized_path = audio_path.replace('.wav', '_optimized.wav')
        librosa.output.write_wav(optimized_path, y, sr)
        
        return optimized_path


# Global audio processor instance
audio_processor = AudioProcessor() 