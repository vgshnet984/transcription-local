import os
import time
import whisper
import torch
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime
import numpy as np
from pathlib import Path
from speakers.simple_identifier import SimpleSpeakerIdentifier
import librosa
from audio.vad_processor import VADProcessor, WEBRTCVAD_AVAILABLE
import re
import warnings
import subprocess
import tempfile

def check_cudnn_installation():
    """Check if cuDNN is properly installed for GPU acceleration"""
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
                return True, path
    
    return False, None

def suggest_cudnn_setup():
    """Provide instructions for cuDNN setup"""
    print("\n" + "="*60)
    print("⚠️  cuDNN NOT FOUND - GPU acceleration may not work")
    print("="*60)
    print("To enable GPU acceleration, install cuDNN:")
    print("1. Run: python scripts/download_cudnn.py")
    print("2. Follow the download instructions")
    print("3. Restart your terminal after installation")
    print("="*60)
    print("Alternatively, use CPU-only mode by setting device='cpu'")
    print("="*60 + "\n")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning)

# Check if pyannote.audio is available
DIARIZATION_AVAILABLE = False
try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    DIARIZATION_AVAILABLE = True
except ImportError:
    pass

# Check if WhisperX is available
WHISPERX_AVAILABLE = False
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    pass

# Check if faster-whisper is available
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    pass

from config import settings
from audio.processor import AudioProcessor

# Tamil romanization to script mapping (simplified)
TAMIL_ROMANIZATION_MAP = {
    'a': 'அ', 'aa': 'ஆ', 'i': 'இ', 'ii': 'ஈ', 'u': 'உ', 'uu': 'ஊ',
    'e': 'எ', 'ee': 'ஏ', 'ai': 'ஐ', 'o': 'ஒ', 'oo': 'ஓ', 'au': 'ஔ',
    'k': 'க', 'ng': 'ங', 'ch': 'ச', 'j': 'ஜ', 'ny': 'ஞ', 't': 'ட',
    'n': 'ன', 'p': 'ப', 'm': 'ம', 'y': 'ய', 'r': 'ர', 'l': 'ல',
    'v': 'வ', 'zh': 'ழ', 'L': 'ள', 'R': 'ற', 'N': 'ண', 'th': 'த',
    's': 'ச', 'h': 'ஹ', 'f': 'ஃப', 'z': 'ஃஜ'
}

# Sanskrit romanization to Devanagari mapping (simplified)
SANSKRIT_ROMANIZATION_MAP = {
    'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ii': 'ई', 'u': 'उ', 'uu': 'ऊ',
    'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ', 'k': 'क', 'kh': 'ख',
    'g': 'ग', 'gh': 'घ', 'ng': 'ङ', 'ch': 'च', 'chh': 'छ', 'j': 'ज',
    'jh': 'झ', 'ny': 'ञ', 't': 'ट', 'th': 'ठ', 'd': 'ड', 'dh': 'ढ',
    'n': 'ण', 'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
    'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व', 'sh': 'श',
    's': 'स', 'h': 'ह'
}

def convert_romanized_to_script(text: str, language: str) -> str:
    """Convert romanized text to native script for Tamil and Sanskrit."""
    if not text:
        return text
    
    mapping = TAMIL_ROMANIZATION_MAP if language == "ta" else SANSKRIT_ROMANIZATION_MAP if language == "sa" else None
    if not mapping:
        return text
    
    converted_text = text
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    for roman in sorted_keys:
        pattern = r'\b' + re.escape(roman) + r'\b'
        converted_text = re.sub(pattern, mapping[roman], converted_text, flags=re.IGNORECASE)
    
    return converted_text

def convert_audio_to_wav(audio_path: str) -> str:
    """Convert audio file to WAV format for better compatibility."""
    try:
        # Check if file is already WAV
        if audio_path.lower().endswith('.wav'):
            return audio_path
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # Use FFmpeg to convert
        cmd = [
            'ffmpeg', '-i', audio_path, 
            '-acodec', 'pcm_s16le', 
            '-ar', '16000', 
            '-ac', '1', 
            '-y',  # Overwrite output file
            temp_wav.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Converted {audio_path} to WAV format")
            return temp_wav.name
        else:
            logger.warning(f"FFmpeg conversion failed: {result.stderr}")
            return audio_path
            
    except Exception as e:
        logger.warning(f"Audio conversion failed: {e}")
        return audio_path

def clean_repetitive_text(text: str, max_repetitions: int = 2) -> str:
    """Remove excessive repetitive text patterns."""
    if not text:
        return text
    
    # Split into sentences and words
    sentences = re.split(r'[.!?]+', text)
    cleaned_sentences = []
    repetition_count = {}
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Count repetitions of this sentence
        if sentence in repetition_count:
            repetition_count[sentence] += 1
        else:
            repetition_count[sentence] = 1
        
        # Only add if not too repetitive
        if repetition_count[sentence] <= max_repetitions:
            cleaned_sentences.append(sentence)
    
    # Join sentences
    result = '. '.join(cleaned_sentences) + ('.' if cleaned_sentences else '')
    
    # Additional cleanup for common repetitive patterns
    # Remove excessive "bye", "thank you", "okay" repetitions
    patterns_to_clean = [
        r'\bbye\b\s*\bbye\b\s*\bbye\b.*',  # Multiple "bye bye bye"
        r'\bthank you\b\s*\bthank you\b\s*\bthank you\b.*',  # Multiple "thank you"
        r'\bokay\b\s*\bokay\b\s*\bokay\b.*',  # Multiple "okay"
    ]
    
    for pattern in patterns_to_clean:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\.+', '.', result)
    result = result.strip()
    
    return result

class TranscriptionEngine:
    """Optimized transcription engine with CUDA acceleration and minimal overhead."""
    
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, 
                 engine: Optional[str] = None, vad_method: Optional[str] = None, 
                 enable_speaker_diarization: Optional[bool] = None, show_romanized_text: bool = False,
                 compute_type: Optional[str] = None, cpu_threads: Optional[int] = None,
                 suppress_logs: bool = True):
        """
        Initialize optimized transcription engine.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device to use (cpu, cuda)
            engine: Transcription engine (whisper, whisperx, faster-whisper)
            vad_method: VAD method (simple, webrtcvad, silero)
            enable_speaker_diarization: Enable speaker diarization
            show_romanized_text: Show romanized text instead of native script
            compute_type: Compute type for faster-whisper (float16, float32, int8)
            cpu_threads: Number of CPU threads for faster-whisper
            suppress_logs: Suppress logs for minimal logging
        """
        # Auto-detect CUDA and optimize settings
        self.device = self._optimize_device(device)
        self.model_size = model_size or self._get_optimal_model_size()
        self.engine = engine or self._get_optimal_engine()
        self.vad_method = vad_method or "none"  # Default to none for speed
        self.enable_speaker_diarization = enable_speaker_diarization if enable_speaker_diarization is not None else False
        self.show_romanized_text = show_romanized_text
        self.compute_type = compute_type or self._get_optimal_compute_type()
        self.cpu_threads = cpu_threads or 4
        self.suppress_logs = suppress_logs
        
        # Initialize components
        self.model = None
        self.whisperx_model = None
        self.faster_whisper_model = None
        self.diarization_pipeline = None
        self.simple_identifier = SimpleSpeakerIdentifier()
        self.vad_processor = VADProcessor(method=self.vad_method)
        self.audio_processor = AudioProcessor()
        
        # Load models with minimal logging
        self._load_model()
        if self.enable_speaker_diarization:
            self._load_diarization()
        
        if not self.suppress_logs:
            logger.info(f"✅ Engine: {self.engine}, Model: {self.model_size}, Device: {self.device}")
    
    def _optimize_device(self, device: Optional[str]) -> str:
        """Optimize device selection for best performance."""
        if device and device != "auto":
            return device
        
        # Auto-detect CUDA
        if torch.cuda.is_available():
            # Check if cuDNN is installed
            cudnn_installed, cudnn_path = check_cudnn_installation()
            
            if not cudnn_installed:
                suggest_cudnn_setup()
                return "cpu"  # Fallback to CPU if cuDNN not found
            
            # Check CUDA memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory >= 8:  # 8GB+ GPU
                return "cuda"
            elif gpu_memory >= 4:  # 4GB+ GPU, use with caution
                return "cuda"
            else:
                return "cpu"
        return "cpu"
    
    def _get_optimal_model_size(self) -> str:
        """Get optimal model size based on device and performance requirements."""
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 12:
                return "large-v3"  # Best accuracy
            elif gpu_memory >= 8:
                return "large"  # Good accuracy
            elif gpu_memory >= 4:
                return "medium"  # Balanced
            else:
                return "small"  # Fast
        else:
            return "base"  # CPU optimized
    
    def _get_optimal_engine(self) -> str:
        """Get optimal engine based on availability and performance."""
        if self.device == "cuda" and FASTER_WHISPER_AVAILABLE:
            return "faster-whisper"  # Best CUDA performance
        elif WHISPERX_AVAILABLE:
            return "whisperx"  # Good accuracy with alignment
        else:
            return "whisper"  # Fallback
    
    def _get_optimal_compute_type(self) -> str:
        """Get optimal compute type for CUDA."""
        if self.device == "cuda":
            # Check if GPU supports float16
            if torch.cuda.is_available():
                return "float16"  # Faster, less memory
            else:
                return "float32"
        return "float32"
    
    def _load_model(self):
        """Load the appropriate transcription model with minimal overhead."""
        try:
            if not self.suppress_logs:
                logger.info(f"📥 Loading {self.engine} model...")
            
            # Load faster-whisper model (best CUDA performance)
            if self.engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                self.faster_whisper_model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    download_root="./models"  # Cache models locally
                )
                if not self.suppress_logs:
                    logger.info(f"faster-whisper model loaded on {self.device}")
                return
            
            # Load WhisperX model (good accuracy with alignment)
            elif self.engine == "whisperx" and WHISPERX_AVAILABLE:
                self.whisperx_model = whisperx.load_model(
                    self.model_size, 
                    self.device,
                    download_root="./models"
                )
                if not self.suppress_logs:
                    logger.info(f"WhisperX model loaded on {self.device}")
                return
            
            # Load standard Whisper model (fallback)
            else:
                self.model = whisper.load_model(
                    self.model_size, 
                    device=self.device,
                    download_root="./models"
                )
                if not self.suppress_logs:
                    logger.info(f"Whisper model loaded on {self.device}")
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            # Fallback to tiny model
            try:
                if not self.suppress_logs:
                    logger.info("Falling back to tiny model")
                if self.engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                    self.faster_whisper_model = FasterWhisperModel("tiny", device=self.device, compute_type=self.compute_type)
                else:
                    self.model = whisper.load_model("tiny", device=self.device)
                if not self.suppress_logs:
                    logger.info("Tiny model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError("No transcription model could be loaded")
    
    def _load_diarization(self):
        """Load speaker diarization pipeline if available."""
        if not DIARIZATION_AVAILABLE:
            return
        
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                return
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            self.diarization_pipeline.to(torch.device(self.device))
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            self.diarization_pipeline = None
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe with optimized performance."""
        start_time = time.time()
        
        if not self.suppress_logs:
            logger.info(f"🎵 Processing: {Path(audio_path).name}")
        
        try:
            # Convert audio to WAV if needed for better compatibility
            processed_audio_path = convert_audio_to_wav(audio_path)
            
            # Run transcription with appropriate engine
            if self.engine == "faster-whisper" and self.faster_whisper_model:
                result = self._transcribe_with_faster_whisper(processed_audio_path, language)
                actual_engine_used = "faster-whisper"
            elif self.engine == "whisperx" and self.whisperx_model:
                result = self._transcribe_with_whisperx(processed_audio_path, language)
                actual_engine_used = "whisperx"
            else:
                result = self._transcribe_with_whisper(processed_audio_path, language)
                actual_engine_used = "whisper"
            
            # Extract text and clean repetitive content
            text = result.get("text", "").strip()
            text = clean_repetitive_text(text, max_repetitions=2)  # Remove excessive repetitions
            
            # Convert script if needed
            detected_language = result.get("language", "en")
            if not self.show_romanized_text:
                if language == "ta" or detected_language == "ta":
                    text = convert_romanized_to_script(text, "ta")
                elif language == "sa" or detected_language == "sa":
                    text = convert_romanized_to_script(text, "sa")
            
            # Get speaker segments (simplified for speed)
            speaker_segments = []
            if self.enable_speaker_diarization and self.diarization_pipeline:
                # Only use diarization if audio is in WAV format
                if processed_audio_path.lower().endswith('.wav'):
                    speaker_segments = self._perform_diarization(processed_audio_path)
                else:
                    logger.warning("Diarization skipped - audio not in WAV format")
            else:
                # Simple speaker identification for speed
                audio_duration = librosa.get_duration(path=processed_audio_path)
                if audio_duration > 60:  # Only for longer audio
                    speaker_segments = self.simple_identifier.identify(processed_audio_path, n_speakers=2)
                else:
                    speaker_segments = self.simple_identifier.identify(processed_audio_path, n_speakers=1)
            
            # Create combined text
            combined_text = self._create_speaker_labeled_text(result, speaker_segments)
            combined_text = clean_repetitive_text(combined_text, max_repetitions=2)  # Clean speaker-labeled text too
            
            transcription_data = {
                "text": combined_text,
                "original_text": text,
                "language": result.get("language", "en"),
                "confidence": self._calculate_confidence(result),
                "segments": result.get("segments", []),
                "speaker_segments": speaker_segments,
                "speakers": self._extract_speaker_info(speaker_segments),
                "processing_time": time.time() - start_time,
                "error": None,
                "actual_engine_used": actual_engine_used,
                "actual_model_used": self.model_size,
                "actual_device_used": self.device
            }
            
            if not self.suppress_logs:
                logger.info(f"✅ Complete: {transcription_data['processing_time']:.1f}s | {len(text)} chars | {actual_engine_used}")
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return {
                "text": "",
                "original_text": "",
                "language": None,
                "confidence": 0.0,
                "segments": [],
                "speaker_segments": [],
                "speakers": [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
        finally:
            # Clean up temporary WAV file if created
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass
    
    def _transcribe_with_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using standard Whisper with optimized settings."""
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
            
        audio = whisper.load_audio(audio_path)
        transcribe_language = language if language and language != "auto" else "en"
        
        # Optimized settings for speed and accuracy
        result = self.model.transcribe(
            audio,
            language=transcribe_language,
            word_timestamps=True,
            verbose=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        return result
    
    def _transcribe_with_whisperx(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using WhisperX with optimized settings."""
        if not WHISPERX_AVAILABLE or self.whisperx_model is None:
            raise RuntimeError("WhisperX not available")
            
        try:
            audio = whisperx.load_audio(audio_path)
            transcribe_language = language if language and language != "auto" else "en"
            
            # Transcribe with WhisperX
            result = self.whisperx_model.transcribe(
                audio,
                language=transcribe_language,
                verbose=False
            )
            
            # Try alignment (skip if fails for speed)
            try:
                model_a, metadata = whisperx.load_align_model(language_code=transcribe_language, device=self.device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, self.device)
            except Exception:
                pass  # Use unaligned result
            
            return result
            
        except Exception as e:
            logger.error(f"WhisperX failed: {e}")
            return self._transcribe_with_whisper(audio_path, language)
    
    def _transcribe_with_faster_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using faster-whisper with optimized settings."""
        if not FASTER_WHISPER_AVAILABLE or self.faster_whisper_model is None:
            raise RuntimeError("faster-whisper not available")
            
        try:
            transcribe_language = language if language and language != "auto" else "en"
            
            # Optimized settings for speed and accuracy
            segments, info = self.faster_whisper_model.transcribe(
                audio_path,
                language=transcribe_language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_filter=False,  # Disable VAD for complete transcription
                chunk_length=30
            )
            
            # Convert to Whisper format
            whisper_segments = []
            full_text = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob if hasattr(segment, 'avg_logprob') else -1.0,
                    "no_speech_prob": segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.0,
                    "words": []
                }
                
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_dict = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability if hasattr(word, 'probability') else 0.0
                        }
                        segment_dict["words"].append(word_dict)
                
                whisper_segments.append(segment_dict)
                full_text += segment.text.strip() + " "
            
            result = {
                "text": full_text.strip(),
                "language": info.language if hasattr(info, 'language') else transcribe_language,
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else 1.0,
                "segments": whisper_segments
            }
            
            return result
            
        except Exception as e:
            logger.error(f"faster-whisper failed: {e}")
            return self._transcribe_with_whisper(audio_path, language)
    
    def _perform_diarization(self, audio_path: str) -> List[Dict]:
        """Perform speaker diarization."""
        try:
            if self.diarization_pipeline is None:
                return []
            
            with ProgressHook() as hook:
                diarization = self.diarization_pipeline(audio_path, hook=hook)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
    
    def _extract_speaker_info(self, speaker_segments: List[Dict]) -> List[Dict]:
        """Extract unique speaker information."""
        speakers = {}
        
        for segment in speaker_segments:
            speaker_id = segment["speaker"]
            duration = segment.get("duration", segment["end"] - segment["start"])
            
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    "speaker_id": speaker_id,
                    "total_duration": 0.0,
                    "segment_count": 0,
                    "first_seen": segment["start"],
                    "last_seen": segment["end"]
                }
            
            speakers[speaker_id]["total_duration"] += duration
            speakers[speaker_id]["segment_count"] += 1
            speakers[speaker_id]["last_seen"] = segment["end"]
        
        return list(speakers.values())
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate overall confidence score."""
        try:
            if "segments" not in result or not result["segments"]:
                return 0.0
            
            confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    conf = np.exp(segment["avg_logprob"])
                    confidences.append(conf)
                elif "confidence" in segment:
                    confidences.append(segment["confidence"])
            
            return float(np.mean(confidences)) if confidences else 0.5
            
        except Exception:
            return 0.5
    
    def _create_speaker_labeled_text(self, whisper_result: Dict, speaker_segments: List[Dict]) -> str:
        """Create combined text with speaker labels."""
        if not speaker_segments or not whisper_result.get("segments"):
            return whisper_result.get("text", "")
        
        whisper_segments = whisper_result["segments"]
        speaker_timeline = []
        
        for segment in speaker_segments:
            speaker_timeline.append({"time": segment["start"], "speaker": segment["speaker"], "type": "start"})
            speaker_timeline.append({"time": segment["end"], "speaker": segment["speaker"], "type": "end"})
        
        speaker_timeline.sort(key=lambda x: x["time"])
        labeled_segments = []
        current_speakers = set()
        
        for whisper_seg in whisper_segments:
            seg_start = whisper_seg["start"]
            seg_end = whisper_seg["end"]
            seg_text = whisper_seg["text"].strip()
            
            active_speakers = set()
            for event in speaker_timeline:
                if event["time"] <= seg_start:
                    if event["type"] == "start":
                        active_speakers.add(event["speaker"])
                    else:
                        active_speakers.discard(event["speaker"])
                elif event["time"] <= seg_end:
                    if event["type"] == "start":
                        active_speakers.add(event["speaker"])
                    else:
                        active_speakers.discard(event["speaker"])
            
            if not active_speakers:
                if current_speakers:
                    active_speakers = current_speakers
                else:
                    active_speakers = {"Speaker 1"}
            
            current_speakers = active_speakers
            speaker_label = f"[{list(active_speakers)[0]}]"
            
            if seg_text:
                labeled_segments.append(f"{speaker_label} {seg_text}")
        
        return " ".join(labeled_segments)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["en", "hi", "te", "ta", "kn", "ml", "gu", "bn", "pa", "ur"]
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        cuda_available = torch.cuda.is_available()
        
        return {
            "current_model": self.model_size,
            "current_device": self.device,
            "current_engine": self.engine,
            "faster_whisper_available": FASTER_WHISPER_AVAILABLE,
            "whisperx_available": WHISPERX_AVAILABLE,
            "diarization_available": DIARIZATION_AVAILABLE,
            "cuda_available": cuda_available,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "optimized_for": "CUDA performance with accuracy"
        }

# Global transcription engine instance
transcription_engine = TranscriptionEngine() 