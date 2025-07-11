import librosa
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import tempfile
import subprocess
import os

# Try to import optional VAD libraries
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logger.warning("webrtcvad not available. Install with: pip install webrtcvad")

try:
    import torch
    import torchaudio
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    logger.warning("Silero VAD not available. Install with: pip install torch torchaudio")


class VADProcessor:
    """Voice Activity Detection processor with multiple methods."""
    
    def __init__(self, method: str = "simple", sample_rate: int = 16000):
        self.method = method
        self.sample_rate = sample_rate
        
        # Initialize VAD methods
        if method == "webrtcvad" and WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        elif method == "silero" and SILERO_AVAILABLE:
            self._load_silero_model()
    
    def _load_silero_model(self):
        """Load Silero VAD model."""
        try:
            if not SILERO_AVAILABLE:
                logger.warning("Silero VAD dependencies not available")
                self.silero_model = None
                self.silero_utils = None
                self.silero_get_speech_timestamps = None
                return
            
            # Load Silero VAD model and utilities
            logger.info("Loading Silero VAD model...")
            model_and_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Handle the returned tuple properly
            if isinstance(model_and_utils, tuple) and len(model_and_utils) == 2:
                self.silero_model, utils = model_and_utils
                # Unpack utils tuple
                if isinstance(utils, tuple) and len(utils) >= 1:
                    self.silero_get_speech_timestamps = utils[0]
                else:
                    self.silero_get_speech_timestamps = None
                self.silero_utils = utils
            else:
                # Fallback if the structure is different
                self.silero_model = model_and_utils
                self.silero_utils = None
                self.silero_get_speech_timestamps = None
                logger.warning("Silero VAD utilities not available, using basic model")
            
            # Move model to appropriate device
            if torch.cuda.is_available():
                self.silero_model = self.silero_model.cuda()
                logger.info("Silero VAD model loaded on CUDA")
            else:
                self.silero_model = self.silero_model.cpu()
                logger.info("Silero VAD model loaded on CPU")
            
            # Set model to evaluation mode
            self.silero_model.eval()
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            self.silero_model = None
            self.silero_utils = None
            self.silero_get_speech_timestamps = None
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Detect voice activity using the specified method.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of voice activity segments
        """
        if self.method == "webrtcvad":
            return self._webrtcvad_detect(audio_path)
        elif self.method == "silero":
            return self._silero_detect(audio_path)
        else:
            return self._simple_detect(audio_path)
    
    def _simple_detect(self, audio_path: str) -> List[Dict[str, Any]]:
        """Simple VAD using energy threshold."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Simple energy-based VAD
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate energy
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Threshold for voice activity
            threshold = np.mean(energy) * 0.5
            
            # Find voice activity segments
            voice_frames = energy > threshold
            
            # Convert to time segments
            segments = []
            start_frame = None
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and start_frame is None:
                    start_frame = i
                elif not is_voice and start_frame is not None:
                    # End of voice segment
                    start_time = start_frame * hop_length / sr
                    end_time = i * hop_length / sr
                    
                    if end_time - start_time > 0.1:  # Minimum 100ms
                        segments.append({
                            "start": start_time,
                            "end": end_time,
                            "duration": end_time - start_time,
                            "confidence": 0.8
                        })
                    start_frame = None
            
            # Handle last segment
            if start_frame is not None:
                start_time = start_frame * hop_length / sr
                end_time = duration
                if end_time - start_time > 0.1:
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "confidence": 0.8
                    })
            
            logger.info(f"Simple VAD detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"Simple VAD failed: {e}")
            return []
    
    def _webrtcvad_detect(self, audio_path: str) -> List[Dict[str, Any]]:
        """WebRTC VAD detection."""
        if not WEBRTCVAD_AVAILABLE:
            logger.warning("webrtcvad not available, falling back to simple VAD")
            return self._simple_detect(audio_path)
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Convert to 16-bit PCM
            y_int16 = (y * 32767).astype(np.int16)
            
            # WebRTC VAD works with 10, 20, or 30ms frames
            frame_duration = 30  # ms
            frame_size = int(sr * frame_duration / 1000)
            
            segments = []
            start_frame = None
            
            for i in range(0, len(y_int16) - frame_size, frame_size):
                frame = y_int16[i:i + frame_size].tobytes()
                is_speech = self.vad.is_speech(frame, sr)
                
                if is_speech and start_frame is None:
                    start_frame = i
                elif not is_speech and start_frame is not None:
                    # End of voice segment
                    start_time = start_frame / sr
                    end_time = i / sr
                    
                    if end_time - start_time > 0.1:  # Minimum 100ms
                        segments.append({
                            "start": start_time,
                            "end": end_time,
                            "duration": end_time - start_time,
                            "confidence": 0.9
                        })
                    start_frame = None
            
            # Handle last segment
            if start_frame is not None:
                start_time = start_frame / sr
                end_time = duration
                if end_time - start_time > 0.1:
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "confidence": 0.9
                    })
            
            logger.info(f"WebRTC VAD detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"WebRTC VAD failed: {e}")
            return self._simple_detect(audio_path)
    
    def _silero_detect(self, audio_path: str) -> List[Dict[str, Any]]:
        """Silero VAD detection."""
        if not SILERO_AVAILABLE:
            logger.warning("Silero VAD not available, falling back to simple VAD")
            return self._simple_detect(audio_path)
        
        if self.silero_model is None:
            logger.warning("Silero VAD model not loaded, falling back to simple VAD")
            return self._simple_detect(audio_path)
        
        if self.silero_get_speech_timestamps is None:
            logger.warning("Silero VAD get_speech_timestamps not available, falling back to simple VAD")
            return self._simple_detect(audio_path)
        
        try:
            # Load audio with torchaudio
            wav, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                sr = 16000
            
            # Ensure correct shape (1, samples)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            
            # Move to appropriate device
            try:
                device = next(self.silero_model.parameters()).device
                wav = wav.to(device)
            except (AttributeError, StopIteration):
                # Fallback to CPU if device detection fails
                wav = wav.cpu()
            
            # Get speech timestamps using Silero VAD
            speech_timestamps = self.silero_get_speech_timestamps(
                wav, 
                self.silero_model, 
                sampling_rate=16000,
                min_speech_duration_ms=250,  # Minimum speech duration
                min_silence_duration_ms=100,  # Minimum silence duration
                speech_pad_ms=30    # Padding around speech segments
            )
            
            # Convert timestamps to segments
            segments = []
            for ts in speech_timestamps:
                start_time = ts['start'] / 16000.0
                end_time = ts['end'] / 16000.0
                duration = end_time - start_time
                
                # Only include segments longer than 100ms
                if duration > 0.1:
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                        "confidence": 0.95  # Silero VAD is very accurate
                    })
            
            logger.info(f"Silero VAD detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"Silero VAD failed: {e}")
            # Fallback to simple VAD on any error
            logger.info("Falling back to simple VAD due to Silero VAD error")
            return self._simple_detect(audio_path) 