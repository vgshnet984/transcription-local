import os
import time
import whisper
import torch
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime
import numpy as np
from pathlib import Path
from src.speakers.simple_identifier import SimpleSpeakerIdentifier
import librosa
from src.audio.vad_processor import VADProcessor, WEBRTCVAD_AVAILABLE
import re

# Check if pyannote.audio is available (lazy import to avoid Windows multiprocessing issues)
DIARIZATION_AVAILABLE = False
try:
    import torch
    # Don't import pyannote.audio at module level on Windows
    import platform
    if platform.system() != 'Windows':
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        DIARIZATION_AVAILABLE = True
    else:
        logger.info("pyannote.audio import deferred on Windows to avoid multiprocessing issues")
except ImportError:
    logger.warning("pyannote.audio not available. Speaker diarization will be disabled.")

# Check if WhisperX is available
WHISPERX_AVAILABLE = False
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    logger.info("WhisperX available for enhanced transcription")
except ImportError:
    logger.info("WhisperX not available, using standard Whisper")

# Check if faster-whisper is available
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisper available for optimized transcription")
except ImportError:
    logger.info("faster-whisper not available, using standard Whisper")

# Check if NVIDIA Parakeet is available
PARAKEET_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForCTC
    PARakeet_AVAILABLE = True
    logger.info("NVIDIA Parakeet available for fast English transcription")
except ImportError:
    logger.info("NVIDIA Parakeet not available, transformers may not be installed")

# Check if NVIDIA NeMo Parakeet is available (commented out due to compilation issues)
PARAKEET_NEMO_AVAILABLE = False
# try:
#     import nemo.collections.asr as nemo_asr
#     PARAKEET_NEMO_AVAILABLE = True
#     logger.info("NVIDIA NeMo Parakeet available for fast English transcription (NeMo)")
# except ImportError:
#     logger.info("NVIDIA NeMo Parakeet not available, install nemo_toolkit[asr]")
logger.info("NVIDIA NeMo Parakeet disabled due to compilation requirements")

# Check if Wav2Vec2 is available (alternative fast ASR)
WAV2VEC2_AVAILABLE = False
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    WAV2VEC2_AVAILABLE = True
    logger.info("Wav2Vec2 available for fast English transcription")
except ImportError:
    logger.info("Wav2Vec2 not available, transformers may not be installed")

from src.config import settings
from src.audio.processor import AudioProcessor

# Tamil romanization to script mapping
TAMIL_ROMANIZATION_MAP = {
    # Vowels
    'a': 'அ', 'aa': 'ஆ', 'A': 'ஆ', 'i': 'இ', 'ii': 'ஈ', 'I': 'ஈ',
    'u': 'உ', 'uu': 'ஊ', 'U': 'ஊ', 'e': 'எ', 'ee': 'ஏ', 'E': 'ஏ',
    'ai': 'ஐ', 'o': 'ஒ', 'oo': 'ஓ', 'O': 'ஓ', 'au': 'ஔ',
    
    # Consonants
    'k': 'க', 'ka': 'க', 'ki': 'கி', 'ku': 'கு', 'ke': 'கெ', 'ko': 'கொ',
    'ng': 'ங', 'nga': 'ங', 'ngi': 'ஙி', 'ngu': 'ஙு', 'nge': 'ஙெ', 'ngo': 'ஙொ',
    'ch': 'ச', 'cha': 'ச', 'chi': 'சி', 'chu': 'சு', 'che': 'செ', 'cho': 'சொ',
    'j': 'ஜ', 'ja': 'ஜ', 'ji': 'ஜி', 'ju': 'ஜு', 'je': 'ஜெ', 'jo': 'ஜொ',
    'ny': 'ஞ', 'nya': 'ஞ', 'nyi': 'ஞி', 'nyu': 'ஞு', 'nye': 'ஞெ', 'nyo': 'ஞொ',
    't': 'ட', 'ta': 'ட', 'ti': 'டி', 'tu': 'டு', 'te': 'டெ', 'to': 'டொ',
    'n': 'ன', 'na': 'ன', 'ni': 'னி', 'nu': 'னு', 'ne': 'னெ', 'no': 'னொ',
    'p': 'ப', 'pa': 'ப', 'pi': 'பி', 'pu': 'பு', 'pe': 'பெ', 'po': 'பொ',
    'm': 'ம', 'ma': 'ம', 'mi': 'மி', 'mu': 'மு', 'me': 'மெ', 'mo': 'மொ',
    'y': 'ய', 'ya': 'ய', 'yi': 'யி', 'yu': 'யு', 'ye': 'யெ', 'yo': 'யொ',
    'r': 'ர', 'ra': 'ர', 'ri': 'ரி', 'ru': 'ரு', 're': 'ரெ', 'ro': 'ரொ',
    'l': 'ல', 'la': 'ல', 'li': 'லி', 'lu': 'லு', 'le': 'லெ', 'lo': 'லொ',
    'v': 'வ', 'va': 'வ', 'vi': 'வி', 'vu': 'வு', 've': 'வெ', 'vo': 'வொ',
    'zh': 'ழ', 'zha': 'ழ', 'zhi': 'ழி', 'zhu': 'ழு', 'zhe': 'ழெ', 'zho': 'ழொ',
    'L': 'ள', 'La': 'ள', 'Li': 'ளி', 'Lu': 'ளு', 'Le': 'ளெ', 'Lo': 'ளொ',
    'R': 'ற', 'Ra': 'ற', 'Ri': 'றி', 'Ru': 'று', 'Re': 'றெ', 'Ro': 'றொ',
    'N': 'ண', 'Na': 'ண', 'Ni': 'ணி', 'Nu': 'ணு', 'Ne': 'ணெ', 'No': 'ணொ',
    'th': 'த', 'tha': 'த', 'thi': 'தி', 'thu': 'து', 'the': 'தெ', 'tho': 'தொ',
    'dh': 'த', 'dha': 'த', 'dhi': 'தி', 'dhu': 'து', 'dhe': 'தெ', 'dho': 'தொ',
    's': 'ச', 'sa': 'ச', 'si': 'சி', 'su': 'சு', 'se': 'செ', 'so': 'சொ',
    'h': 'ஹ', 'ha': 'ஹ', 'hi': 'ஹி', 'hu': 'ஹு', 'he': 'ஹெ', 'ho': 'ஹொ',
    'f': 'ஃப', 'fa': 'ஃப', 'fi': 'ஃபி', 'fu': 'ஃபு', 'fe': 'ஃபெ', 'fo': 'ஃபொ',
    'z': 'ஃஜ', 'za': 'ஃஜ', 'zi': 'ஃஜி', 'zu': 'ஃஜு', 'ze': 'ஃஜெ', 'zo': 'ஃஜொ',
    
    # Common words and phrases
    'vanakkam': 'வணக்கம்',
    'nalam': 'நலம்',
    'nalamariya': 'நலமறிய',
    'aval': 'அவள்',
    'ranjani': 'ரஞ்சனி',
    'en': 'என்',
    'mane': 'மனை',
    'manevi': 'மனைவி',
    'magal': 'மகள்',
    'magalgal': 'மகள்கள்',
    'shamathmika': 'சமத்மிகா',
    'varnika': 'வர்ணிகா',
    'vayadhu': 'வயது',
    'padimundru': 'பதின்மூன்று',
    'ezu': 'ஏழு',
    
    # Numbers
    'onru': 'ஒன்று', 'iru': 'இரு', 'moonru': 'மூன்று', 'naalu': 'நான்கு',
    'ainthu': 'ஐந்து', 'aaru': 'ஆறு', 'ezhu': 'ஏழு', 'ettu': 'எட்டு',
    'onpathu': 'ஒன்பது', 'pathu': 'பத்து'
}

# Sanskrit romanization to Devanagari mapping
SANSKRIT_ROMANIZATION_MAP = {
    # Vowels
    'a': 'अ', 'aa': 'आ', 'A': 'आ', 'i': 'इ', 'ii': 'ई', 'I': 'ई',
    'u': 'उ', 'uu': 'ऊ', 'U': 'ऊ', 'e': 'ए', 'ee': 'ए', 'E': 'ए',
    'ai': 'ऐ', 'o': 'ओ', 'oo': 'ओ', 'O': 'ओ', 'au': 'औ',
    
    # Consonants
    'k': 'क', 'ka': 'क', 'ki': 'कि', 'ku': 'कु', 'ke': 'के', 'ko': 'को',
    'kh': 'ख', 'kha': 'ख', 'khi': 'खि', 'khu': 'खु', 'khe': 'खे', 'kho': 'खो',
    'g': 'ग', 'ga': 'ग', 'gi': 'गि', 'gu': 'गु', 'ge': 'गे', 'go': 'गो',
    'gh': 'घ', 'gha': 'घ', 'ghi': 'घि', 'ghu': 'घु', 'ghe': 'घे', 'gho': 'घो',
    'ng': 'ङ', 'nga': 'ङ', 'ngi': 'ङि', 'ngu': 'ङु', 'nge': 'ङे', 'ngo': 'ङो',
    'ch': 'च', 'cha': 'च', 'chi': 'चि', 'chu': 'चु', 'che': 'चे', 'cho': 'चो',
    'chh': 'छ', 'chha': 'छ', 'chhi': 'छि', 'chhu': 'छु', 'chhe': 'छे', 'chho': 'छो',
    'j': 'ज', 'ja': 'ज', 'ji': 'जि', 'ju': 'जु', 'je': 'जे', 'jo': 'जो',
    'jh': 'झ', 'jha': 'झ', 'jhi': 'झि', 'jhu': 'झु', 'jhe': 'झे', 'jho': 'झो',
    'ny': 'ञ', 'nya': 'ञ', 'nyi': 'ञि', 'nyu': 'ञु', 'nye': 'ञे', 'nyo': 'ञो',
    't': 'ट', 'ta': 'ट', 'ti': 'टि', 'tu': 'टु', 'te': 'टे', 'to': 'टो',
    'th': 'ठ', 'tha': 'ठ', 'thi': 'ठि', 'thu': 'ठु', 'the': 'ठे', 'tho': 'ठो',
    'd': 'ड', 'da': 'ड', 'di': 'डि', 'du': 'डु', 'de': 'डे', 'do': 'डो',
    'dh': 'ढ', 'dha': 'ढ', 'dhi': 'ढि', 'dhu': 'ढु', 'dhe': 'ढे', 'dho': 'ढो',
    'n': 'ण', 'na': 'ण', 'ni': 'णि', 'nu': 'णु', 'ne': 'णे', 'no': 'णो',
    'p': 'प', 'pa': 'प', 'pi': 'पि', 'pu': 'पु', 'pe': 'पे', 'po': 'पो',
    'ph': 'फ', 'pha': 'फ', 'phi': 'फि', 'phu': 'फु', 'phe': 'फे', 'pho': 'फो',
    'b': 'ब', 'ba': 'ब', 'bi': 'बि', 'bu': 'बु', 'be': 'बे', 'bo': 'बो',
    'bh': 'भ', 'bha': 'भ', 'bhi': 'भि', 'bhu': 'भु', 'bhe': 'भे', 'bho': 'भो',
    'm': 'म', 'ma': 'म', 'mi': 'मि', 'mu': 'मु', 'me': 'मे', 'mo': 'मो',
    'y': 'य', 'ya': 'य', 'yi': 'यि', 'yu': 'यु', 'ye': 'ये', 'yo': 'यो',
    'r': 'र', 'ra': 'र', 'ri': 'रि', 'ru': 'रु', 're': 'रे', 'ro': 'रो',
    'l': 'ल', 'la': 'ल', 'li': 'लि', 'lu': 'लु', 'le': 'ले', 'lo': 'लो',
    'v': 'व', 'va': 'व', 'vi': 'वि', 'vu': 'वु', 've': 'वे', 'vo': 'वो',
    'w': 'व', 'wa': 'व', 'wi': 'वि', 'wu': 'वु', 'we': 'वे', 'wo': 'वो',
    'sh': 'श', 'sha': 'श', 'shi': 'शि', 'shu': 'शु', 'she': 'शे', 'sho': 'शो',
    's': 'स', 'sa': 'स', 'si': 'सि', 'su': 'सु', 'se': 'से', 'so': 'सो',
    'h': 'ह', 'ha': 'ह', 'hi': 'हि', 'hu': 'हु', 'he': 'हे', 'ho': 'हो',
    
    # Common Sanskrit words
    'namaste': 'नमस्ते',
    'om': 'ॐ',
    'shanti': 'शान्ति',
    'dharma': 'धर्म',
    'karma': 'कर्म',
    'yoga': 'योग',
    'veda': 'वेद',
    'mantra': 'मन्त्र',
    'guru': 'गुरु',
    'deva': 'देव',
    'devi': 'देवी',
    'brahma': 'ब्रह्म',
    'vishnu': 'विष्णु',
    'shiva': 'शिव',
    'ganesha': 'गणेश',
    'lakshmi': 'लक्ष्मी',
    'saraswati': 'सरस्वती',
    'krishna': 'कृष्ण',
    'rama': 'राम',
    'sita': 'सीता',
    'hanuman': 'हनुमान',
    'durga': 'दुर्गा',
    'kali': 'काली',
    'parvati': 'पार्वती',
    'ganesh': 'गणेश',
    'shakti': 'शक्ति',
    'atman': 'आत्मन्',
    'moksha': 'मोक्ष',
    'samsara': 'संसार',
    'nirvana': 'निर्वाण',
    'buddha': 'बुद्ध',
    'buddhism': 'बौद्धधर्म',
    'ahimsa': 'अहिंसा',
    'satya': 'सत्य',
    'ahimsa': 'अहिंसा',
    'brahmacharya': 'ब्रह्मचर्य',
    'aparigraha': 'अपरिग्रह',
    'ishvara': 'ईश्वर',
    'prana': 'प्राण',
    'chakra': 'चक्र',
    'kundalini': 'कुण्डलिनी',
    'samadhi': 'समाधि',
    'dhyana': 'ध्यान',
    'pranayama': 'प्राणायाम',
    'asana': 'आसन',
    'sutra': 'सूत्र',
    'upanishad': 'उपनिषद्',
    'purana': 'पुराण',
    'itihasa': 'इतिहास',
    'shastra': 'शास्त्र',
    'vedanta': 'वेदान्त',
    'sankhya': 'सांख्य',
    'nyaya': 'न्याय',
    'vaisheshika': 'वैशेषिक',
    'mimamsa': 'मीमांसा',
    'advaita': 'अद्वैत',
    'dvaita': 'द्वैत',
    'vishishtadvaita': 'विशिष्टाद्वैत'
}

def convert_romanized_to_script(text: str, language: str) -> str:
    """
    Convert romanized text to native script for Tamil and Sanskrit.
    
    Args:
        text: Romanized text
        language: Language code ('ta' for Tamil, 'sa' for Sanskrit)
    
    Returns:
        Text in native script
    """
    if not text:
        return text
    
    # Choose the appropriate mapping
    if language == "ta":
        mapping = TAMIL_ROMANIZATION_MAP
    elif language == "sa":
        mapping = SANSKRIT_ROMANIZATION_MAP
    else:
        return text  # No conversion for other languages
    
    # Convert the text
    converted_text = text
    
    # Sort by length (longest first) to avoid partial matches
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    for roman in sorted_keys:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(roman) + r'\b'
        converted_text = re.sub(pattern, mapping[roman], converted_text, flags=re.IGNORECASE)
    
    return converted_text

class TranscriptionEngine:
    """Enhanced transcription engine with support for multiple engines and VAD methods."""
    
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, 
                 engine: Optional[str] = None, vad_method: Optional[str] = None, 
                 enable_speaker_diarization: Optional[bool] = None, show_romanized_text: bool = False,
                 compute_type: Optional[str] = None, cpu_threads: Optional[int] = None):
        """
        Initialize transcription engine.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device to use (cpu, cuda)
            engine: Transcription engine (whisper, whisperx, faster-whisper)
            vad_method: VAD method (simple, webrtcvad, silero)
            enable_speaker_diarization: Enable speaker diarization (overrides global setting)
            show_romanized_text: Show romanized text instead of native script
            compute_type: Compute type for faster-whisper (float16, float32, int8)
            cpu_threads: Number of CPU threads for faster-whisper
        """
        # Use provided values or fall back to settings
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.device
        self.engine = engine or settings.transcription_engine
        self.vad_method = vad_method or settings.vad_method
        self.enable_speaker_diarization = enable_speaker_diarization if enable_speaker_diarization is not None else settings.enable_speaker_diarization
        self.show_romanized_text = show_romanized_text
        self.compute_type = compute_type or "float16" if self.device == "cuda" else "float32"
        self.cpu_threads = cpu_threads or 4
        
        # Initialize components
        self.model = None
        self.whisperx_model = None
        self.faster_whisper_model = None
        self.parakeet_model = None
        self.parakeet_processor = None
        self.diarization_pipeline = None
        self.simple_identifier = SimpleSpeakerIdentifier()
        self.vad_processor = VADProcessor(method=self.vad_method)
        self.audio_processor = AudioProcessor()
        self.parakeet_nemo_model = None
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        
        # Load models
        self._load_model()
        self._load_diarization()
        
        logger.info(f"TranscriptionEngine initialized with {self.engine} engine, {self.vad_method} VAD")
    
    def _load_model(self):
        """Load the appropriate transcription model based on engine setting."""
        try:
            # Load NVIDIA NeMo Parakeet model if requested and available
            if self.engine == "parakeet-nemo" and PARAKEET_NEMO_AVAILABLE:
                logger.info("Loading NVIDIA NeMo Parakeet model for fast English transcription")
                # self.parakeet_nemo_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
                logger.warning("NVIDIA NeMo Parakeet not available due to compilation requirements")
                return
            elif self.engine == "parakeet-nemo" and not PARAKEET_NEMO_AVAILABLE:
                logger.warning("NVIDIA NeMo Parakeet requested but not available, falling back to standard Whisper")
            
            # Load NVIDIA Parakeet model if requested and available
            if self.engine == "parakeet" and PARakeet_AVAILABLE:
                logger.info("Loading NVIDIA Parakeet model for fast English transcription")
                logger.info(f"Device: {self.device}")
                
                # Load Parakeet model and processor
                model_name = "nvidia/parakeet-tdt-0.6b-v2"
                self.parakeet_processor = AutoProcessor.from_pretrained(model_name)
                self.parakeet_model = AutoModelForCTC.from_pretrained(model_name)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    self.parakeet_model = self.parakeet_model.to("cuda")
                
                logger.info(f"NVIDIA Parakeet model loaded successfully on {self.device}")
                return
            elif self.engine == "parakeet" and not PARakeet_AVAILABLE:
                logger.warning("NVIDIA Parakeet requested but not available, falling back to standard Whisper")
            
            # Load Wav2Vec2 model if requested and available
            if self.engine == "wav2vec2" and WAV2VEC2_AVAILABLE:
                logger.info("Loading Wav2Vec2 model for fast English transcription")
                logger.info(f"Device: {self.device}")
                
                # Load Wav2Vec2 model and processor
                model_name = "facebook/wav2vec2-large-960h"
                self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    self.wav2vec2_model = self.wav2vec2_model.cuda()
                
                logger.info(f"Wav2Vec2 model loaded successfully on {self.device}")
                return
            elif self.engine == "wav2vec2" and not WAV2VEC2_AVAILABLE:
                logger.warning("Wav2Vec2 requested but not available, falling back to standard Whisper")
            
            # Load faster-whisper model if requested and available
            if self.engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                logger.info(f"Loading faster-whisper model: {self.model_size}")
                logger.info(f"Device: {self.device}, Compute type: {self.compute_type}, CPU threads: {self.cpu_threads}")
                
                # Add progress message for large models
                if self.model_size in ["large", "large-v2", "large-v3"]:
                    logger.info("Loading large model - this may take several minutes on CPU...")
                
                self.faster_whisper_model = FasterWhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads
                )
                logger.info(f"faster-whisper model loaded successfully on {self.device}")
                return
            elif self.engine == "faster-whisper" and not FASTER_WHISPER_AVAILABLE:
                logger.warning("faster-whisper requested but not available, falling back to standard Whisper")
            
            # Load WhisperX model if requested and available
            if self.engine == "whisperx" and WHISPERX_AVAILABLE:
                logger.info(f"Loading WhisperX model: {self.model_size}")
                self.whisperx_model = whisperx.load_model(self.model_size, self.device)
                logger.info(f"WhisperX model loaded successfully on {self.device}")
                return
            elif self.engine == "whisperx" and not WHISPERX_AVAILABLE:
                logger.warning("WhisperX requested but not available, using standard Whisper")
            
            # Always load Whisper model as fallback
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Add progress messages for large models
            if self.model_size in ["large", "large-v2", "large-v3"]:
                logger.info("Loading large Whisper model - this may take several minutes on CPU...")
                logger.info("For faster loading, consider using 'faster-whisper' engine or smaller models")
            elif self.model_size in ["medium", "medium.en"]:
                logger.info("Loading medium Whisper model - this may take a minute on CPU...")
            
            if self.model_size == "large-v3":
                logger.info("Using standard Whisper with large model for enhanced accuracy")
            
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model loaded successfully on {self.device}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to tiny model
            try:
                logger.info("Falling back to tiny model")
                if self.engine == "parakeet" and PARakeet_AVAILABLE:
                    # Parakeet doesn't have different sizes, use standard fallback
                    self.model = whisper.load_model("tiny", device=self.device)
                elif self.engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                    self.faster_whisper_model = FasterWhisperModel("tiny", device=self.device, compute_type=self.compute_type)
                else:
                    self.model = whisper.load_model("tiny", device=self.device)
                logger.info("Tiny model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError("No transcription model could be loaded")
    
    def _load_diarization(self):
        """Load speaker diarization pipeline if available."""
        if not DIARIZATION_AVAILABLE:
            logger.info("Speaker diarization not available (pyannote.audio not installed)")
            return
        
        try:
            # Check if we have the HuggingFace token for pyannote
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN not set. Speaker diarization will be disabled.")
                return
            
            # Lazy import for Windows compatibility
            import platform
            if platform.system() == 'Windows':
                try:
                    from pyannote.audio import Pipeline
                    from pyannote.audio.pipelines.utils.hook import ProgressHook
                except ImportError:
                    logger.warning("pyannote.audio not available on Windows. Speaker diarization will be disabled.")
                    return
            else:
                from pyannote.audio import Pipeline
                from pyannote.audio.pipelines.utils.hook import ProgressHook
            
            logger.info("Loading speaker diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            self.diarization_pipeline.to(torch.device(self.device))
            logger.info("Speaker diarization pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            self.diarization_pipeline = None
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio file with speaker identification.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, defaults to English for Indian accents)
            
        Returns:
            Dict containing transcription results with speaker labels in text
        """
        start_time = time.time()
        preprocessed_path = None
        
        # Add language-specific recommendations
        if language and language.lower() in ["tamil", "ta"]:
            if self.model_size in ["large", "large-v2", "large-v3"] and self.engine == "whisper":
                logger.info("Tamil language detected with large Whisper model")
                logger.info("Recommendation: Use 'faster-whisper' engine or smaller models for faster processing")
        
        try:
            # Preprocess audio if enabled
            if settings.enable_audio_preprocessing:
                logger.info("Preprocessing audio for better transcription...")
                preprocessed_path = self.audio_processor.preprocess_audio(audio_path)
                if preprocessed_path:
                    logger.info(f"Audio preprocessed and saved to: {preprocessed_path}")
                else:
                    logger.warning("Audio preprocessing failed, using original file")
                    preprocessed_path = audio_path
            else:
                preprocessed_path = audio_path
            
            # Run transcription with appropriate engine
            logger.info(f"Running {self.engine} transcription...")
            
            actual_engine_used = "whisper"  # Default
            actual_model_used = self.model_size  # Default
            
            if self.engine == "parakeet-nemo" and PARAKEET_NEMO_AVAILABLE and self.parakeet_nemo_model is not None:
                try:
                    result = self._transcribe_with_parakeet_nemo(preprocessed_path, language)
                    actual_engine_used = "parakeet-nemo"
                    actual_model_used = "parakeet-tdt-0.6b-v2-nemo"
                except Exception as e:
                    logger.warning(f"NVIDIA NeMo Parakeet failed, falling back to Whisper: {e}")
                    result = self._transcribe_with_whisper(preprocessed_path, language)
                    actual_engine_used = "whisper"
                    actual_model_used = self.model_size
            elif self.engine == "parakeet" and PARakeet_AVAILABLE and self.parakeet_model is not None:
                # Use NVIDIA Parakeet for fast English transcription
                try:
                    result = self._transcribe_with_parakeet(preprocessed_path, language)
                    actual_engine_used = "parakeet"
                    actual_model_used = "parakeet-tdt-0.6b-v2"
                except Exception as e:
                    logger.warning(f"NVIDIA Parakeet failed, falling back to Whisper: {e}")
                    result = self._transcribe_with_whisper(preprocessed_path, language)
                    actual_engine_used = "whisper"
                    actual_model_used = self.model_size
            elif self.engine == "faster-whisper" and FASTER_WHISPER_AVAILABLE and self.faster_whisper_model is not None:
                # Use faster-whisper for optimized transcription
                try:
                    result = self._transcribe_with_faster_whisper(preprocessed_path, language)
                    actual_engine_used = "faster-whisper"
                    actual_model_used = self.model_size
                except Exception as e:
                    logger.warning(f"faster-whisper failed, falling back to Whisper: {e}")
                    result = self._transcribe_with_whisper(preprocessed_path, language)
                    actual_engine_used = "whisper"
                    actual_model_used = self.model_size
            elif self.engine == "wav2vec2" and WAV2VEC2_AVAILABLE and self.wav2vec2_model is not None:
                # Use Wav2Vec2 for fast English transcription
                try:
                    result = self._transcribe_with_wav2vec2(preprocessed_path, language)
                    actual_engine_used = "wav2vec2"
                    actual_model_used = "wav2vec2-large-960h"
                except Exception as e:
                    logger.warning(f"Wav2Vec2 failed, falling back to Whisper: {e}")
                    result = self._transcribe_with_whisper(preprocessed_path, language)
                    actual_engine_used = "whisper"
                    actual_model_used = self.model_size
            elif self.engine == "whisperx" and WHISPERX_AVAILABLE and self.whisperx_model is not None:
                # Use WhisperX for enhanced transcription
                try:
                    result = self._transcribe_with_whisperx(preprocessed_path, language)
                    actual_engine_used = "whisperx"
                    actual_model_used = getattr(self, 'whisperx_model_size', "large-v3")
                except Exception as e:
                    logger.warning(f"WhisperX failed, falling back to Whisper: {e}")
                    result = self._transcribe_with_whisper(preprocessed_path, language)
                    actual_engine_used = "whisper"
                    actual_model_used = self.model_size
            else:
                # Use standard Whisper
                result = self._transcribe_with_whisper(preprocessed_path, language)
                actual_engine_used = "whisper"
                actual_model_used = self.model_size
            
            # Extract basic transcription info
            if "text" in result:
                text = result["text"]
                if isinstance(text, str):
                    text = text.strip()
                else:
                    text = str(text).strip()
            else:
                # For WhisperX aligned results, reconstruct text from segments
                text = " ".join([segment.get("text", "") for segment in result.get("segments", [])]).strip()
            
            # Convert romanized Tamil to Tamil script if Tamil language is detected/selected
            detected_language = result.get("language", "en")
            logger.info(f"show_romanized_text: {self.show_romanized_text}, language: {language}, detected_language: {detected_language}")
            
            if not self.show_romanized_text:  # Only convert if user wants native script
                if language == "ta" or detected_language == "ta":
                    logger.info("Converting romanized Tamil to Tamil script")
                    text = convert_romanized_to_script(text, "ta")
                    
                    # Also convert segments
                    if "segments" in result:
                        for segment in result["segments"]:
                            if "text" in segment:
                                segment["text"] = convert_romanized_to_script(segment["text"], "ta")
                
                # Convert romanized Sanskrit to Devanagari script if Sanskrit language is detected/selected
                elif language == "sa" or detected_language == "sa":
                    logger.info("Converting romanized Sanskrit to Devanagari script")
                    text = convert_romanized_to_script(text, "sa")
                    
                    # Also convert segments
                    if "segments" in result:
                        for segment in result["segments"]:
                            if "text" in segment:
                                segment["text"] = convert_romanized_to_script(segment["text"], "sa")
            else:
                logger.info("Keeping romanized text as requested by user")
            
            # Get speaker segments
            if self.diarization_pipeline and self.enable_speaker_diarization:
                logger.info("Running speaker diarization...")
                speaker_segments = self._perform_diarization(preprocessed_path)
            else:
                # Use VAD-based speaker identification
                logger.info(f"Using {self.vad_method} VAD for speaker identification...")
                # For most cases, assume single speaker unless audio is very long
                audio_duration = librosa.get_duration(path=preprocessed_path)
                if audio_duration > 120:  # Only try multiple speakers for very long audio
                    speaker_segments = self.simple_identifier.identify(preprocessed_path, n_speakers=2)
                else:
                    speaker_segments = self.simple_identifier.identify(preprocessed_path, n_speakers=1)
            
            # Create combined text with speaker labels
            combined_text = self._create_speaker_labeled_text(result, speaker_segments)
            
            transcription_data = {
                "text": combined_text,  # Use combined text with speaker labels
                "original_text": text,  # Keep original text without labels
                "language": result.get("language", "en"),
                "confidence": self._calculate_confidence(result),
                "segments": result.get("segments", []),
                "speaker_segments": speaker_segments,
                "speakers": self._extract_speaker_info(speaker_segments),
                "processing_time": time.time() - start_time,
                "error": None,
                "actual_engine_used": actual_engine_used,
                "actual_model_used": actual_model_used,
                "actual_device_used": self.device
            }
            
            logger.info(f"Transcription completed in {transcription_data['processing_time']:.2f}s")
            return transcription_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
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
            # Clean up preprocessed file if it's different from original
            if preprocessed_path and preprocessed_path != audio_path:
                self.audio_processor.cleanup_file(preprocessed_path)
    
    def _transcribe_with_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using standard Whisper with enhanced settings for Indian accents."""
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
            
        # Load audio
        audio = whisper.load_audio(audio_path)
        
        # Use the specified language or default to English
        transcribe_language = language if language and language != "auto" else "en"
        
        # Enhanced settings for better accuracy with accents
        result = self.model.transcribe(
            audio,
            language=transcribe_language,
            word_timestamps=True,
            verbose=False,
            temperature=0.0,  # Lower temperature for more consistent output
            compression_ratio_threshold=2.4,  # More permissive for accents
            logprob_threshold=-1.0,  # More permissive for accents
            no_speech_threshold=0.6  # More permissive for speech detection
        )
        
        return result
    
    def _transcribe_with_whisperx(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using WhisperX for enhanced quality."""
        if not WHISPERX_AVAILABLE:
            raise RuntimeError("WhisperX not available")
        
        if self.whisperx_model is None:
            raise RuntimeError("WhisperX model not loaded")
            
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Use the specified language or default to English
            transcribe_language = language if language and language != "auto" else "en"
            
            # Transcribe with WhisperX - force the language to prevent auto-detection
            result = self.whisperx_model.transcribe(
                audio,
                language=transcribe_language,
                verbose=False
            )
            
            # Ensure the detected language matches what we specified
            detected_language = result.get("language", transcribe_language)
            if detected_language != transcribe_language:
                logger.warning(f"WhisperX detected language '{detected_language}' but we specified '{transcribe_language}'. Forcing specified language.")
                result["language"] = transcribe_language
            
            # Try to align timestamps, but skip if no alignment model is available
            try:
                model_a, metadata = whisperx.load_align_model(language_code=transcribe_language, device=self.device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, self.device)
                logger.info(f"WhisperX alignment completed for language: {transcribe_language}")
            except Exception as align_error:
                logger.warning(f"Alignment failed for language {transcribe_language}, using unaligned result: {align_error}")
                # Keep the unaligned result - it's still better than falling back to Whisper
            
            return result
            
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            # Fallback to standard Whisper
            logger.info("Falling back to standard Whisper")
            return self._transcribe_with_whisper(audio_path, language)
    
    def _transcribe_with_parakeet(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using NVIDIA Parakeet for fast English transcription."""
        if not PARakeet_AVAILABLE:
            raise RuntimeError("NVIDIA Parakeet not available")
        
        if self.parakeet_model is None or self.parakeet_processor is None:
            raise RuntimeError("NVIDIA Parakeet model not loaded")
            
        try:
            import librosa
            import torch
            
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Process audio with Parakeet processor
            inputs = self.parakeet_processor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # Move to device if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.parakeet_model(**inputs).logits
            
            # Get predicted token ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode the audio
            transcription = self.parakeet_processor.batch_decode(predicted_ids)
            text = transcription[0].strip()
            
            # Create segments (Parakeet doesn't provide timestamps, so we create a single segment)
            segments = [{
                "start": 0.0,
                "end": librosa.get_duration(path=audio_path),
                "text": text,
                "avg_logprob": -1.0,
                "no_speech_prob": 0.0,
                "words": []
            }]
            
            # Create result in Whisper format
            result = {
                "text": text,
                "language": "en",  # Parakeet is English-only
                "language_probability": 1.0,
                "segments": segments
            }
            
            logger.info(f"NVIDIA Parakeet transcription completed")
            return result
            
        except Exception as e:
            logger.error(f"NVIDIA Parakeet transcription failed: {e}")
            # Fallback to standard Whisper
            logger.info("Falling back to standard Whisper")
            return self._transcribe_with_whisper(audio_path, language)
    
    def _transcribe_with_faster_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using faster-whisper for optimized performance."""
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not available")
        
        if self.faster_whisper_model is None:
            raise RuntimeError("faster-whisper model not loaded")
            
        try:
            # Use the specified language or default to English
            transcribe_language = language if language and language != "auto" else "en"
            
            # Transcribe with faster-whisper - optimized settings
            segments, info = self.faster_whisper_model.transcribe(
                audio_path,
                language=transcribe_language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert faster-whisper segments to Whisper format
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
                
                # Add word-level timestamps if available
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
            
            # Create result in Whisper format
            result = {
                "text": full_text.strip(),
                "language": info.language if hasattr(info, 'language') else transcribe_language,
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else 1.0,
                "segments": whisper_segments
            }
            
            logger.info(f"faster-whisper transcription completed for language: {transcribe_language}")
            return result
            
        except Exception as e:
            logger.error(f"faster-whisper transcription failed: {e}")
            # Fallback to standard Whisper
            logger.info("Falling back to standard Whisper")
            return self._transcribe_with_whisper(audio_path, language)
    
    def _transcribe_with_parakeet_nemo(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using NVIDIA NeMo Parakeet for fast English transcription."""
        if not PARAKEET_NEMO_AVAILABLE:
            raise RuntimeError("NVIDIA NeMo Parakeet not available")
        if self.parakeet_nemo_model is None:
            raise RuntimeError("NVIDIA NeMo Parakeet model not loaded")
        try:
            transcriptions = self.parakeet_nemo_model.transcribe([audio_path])
            text = transcriptions[0].strip() if transcriptions else ""
            duration = None
            try:
                import librosa
                duration = librosa.get_duration(path=audio_path)
            except Exception:
                duration = None
            segments = [{
                "start": 0.0,
                "end": duration if duration else 0.0,
                "text": text,
                "avg_logprob": -1.0,
                "no_speech_prob": 0.0,
                "words": []
            }]
            result = {
                "text": text,
                "language": "en",
                "language_probability": 1.0,
                "segments": segments
            }
            logger.info("NVIDIA NeMo Parakeet transcription completed")
            return result
        except Exception as e:
            logger.error(f"NVIDIA NeMo Parakeet transcription failed: {e}")
            logger.info("Falling back to standard Whisper")
            return self._transcribe_with_whisper(audio_path, language)
    
    def _transcribe_with_wav2vec2(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe using Wav2Vec2 for fast English transcription."""
        if not WAV2VEC2_AVAILABLE:
            raise RuntimeError("Wav2Vec2 not available")
        if self.wav2vec2_model is None or self.wav2vec2_processor is None:
            raise RuntimeError("Wav2Vec2 model not loaded")
        
        try:
            import librosa
            import torch
            
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio with Wav2Vec2
            inputs = self.wav2vec2_processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Move inputs to device
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)
            text = transcription[0].strip()
            
            # Get duration
            duration = librosa.get_duration(path=audio_path)
            
            # Create segments (Wav2Vec2 doesn't provide timestamps, so we create one segment)
            segments = [{
                "start": 0.0,
                "end": duration,
                "text": text,
                "avg_logprob": -1.0,
                "no_speech_prob": 0.0,
                "words": []
            }]
            
            result = {
                "text": text,
                "language": "en",  # Wav2Vec2 is English-only
                "language_probability": 1.0,
                "segments": segments
            }
            
            logger.info("Wav2Vec2 transcription completed")
            return result
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            logger.info("Falling back to standard Whisper")
            return self._transcribe_with_whisper(audio_path, language)
    
    def transcribe_batch(self, audio_paths: List[str], language: Optional[str] = None,
                        batch_size: int = 4, max_workers: int = 2) -> List[Dict]:
        """
        Transcribe multiple audio files with optimized batch processing.
        
        Args:
            audio_paths: List of audio file paths
            language: Language code (optional)
            batch_size: Number of files to process in each batch
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of transcription results
        """
        import concurrent.futures
        from itertools import islice
        
        results = []
        total_files = len(audio_paths)
        
        logger.info(f"Starting batch transcription of {total_files} files with batch_size={batch_size}, max_workers={max_workers}")
        
        # Process files in batches
        for i in range(0, total_files, batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_files + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_paths)} files")
            
            # Process batch with parallel workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit transcription tasks
                future_to_path = {
                    executor.submit(self.transcribe, path, language): path 
                    for path in batch_paths
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        result['file_path'] = path
                        results.append(result)
                        logger.info(f"Completed: {Path(path).name}")
                    except Exception as e:
                        logger.error(f"Failed to transcribe {path}: {e}")
                        results.append({
                            'file_path': path,
                            'text': '',
                            'error': str(e),
                            'processing_time': 0.0
                        })
            
            # Clear GPU memory between batches if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory after batch")
        
        logger.info(f"Batch transcription completed. Processed {len(results)} files")
        return results
    
    def _perform_diarization(self, audio_path: str) -> List[Dict]:
        """Perform speaker diarization on audio file."""
        try:
            if self.diarization_pipeline is None:
                return []
                
            with ProgressHook() as hook:
                diarization = self.diarization_pipeline(audio_path, hook=hook)
            
            # Convert diarization to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                })
            
            logger.info(f"Diarization found {len(segments)} speaker segments")
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
    
    def _extract_speaker_info(self, speaker_segments: List[Dict]) -> List[Dict]:
        """Extract unique speaker information from segments."""
        speakers = {}
        
        for segment in speaker_segments:
            speaker_id = segment["speaker"]
            # Calculate duration if not provided
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
        """Calculate overall confidence score from transcription result."""
        try:
            if "segments" not in result or not result["segments"]:
                return 0.0
            
            # Calculate average confidence from segments
            confidences = []
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    # Convert log probability to confidence
                    conf = np.exp(segment["avg_logprob"])
                    confidences.append(conf)
                elif "confidence" in segment:
                    confidences.append(segment["confidence"])
            
            if confidences:
                return float(np.mean(confidences))
            else:
                return 0.5  # Default confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["en", "hi", "te", "ta", "kn", "ml", "gu", "bn", "pa", "ur"]  # English + Indian languages
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        # Check CUDA availability
        cuda_available = torch.cuda.is_available() if torch.cuda.is_available() else False
        
        return {
            "current_model": self.model_size,
            "current_device": self.device,
            "current_engine": self.engine,
            "faster_whisper_available": FASTER_WHISPER_AVAILABLE,
            "whisperx_available": WHISPERX_AVAILABLE,
            "parakeet_available": PARakeet_AVAILABLE,
            "parakeet_nemo_available": PARAKEET_NEMO_AVAILABLE,
            "diarization_available": DIARIZATION_AVAILABLE,
            "webrtcvad_available": WEBRTCVAD_AVAILABLE,
            "cuda_available": cuda_available,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "audio_preprocessing": settings.enable_audio_preprocessing,
            "vad_enabled": settings.enable_vad,
            "denoising_enabled": settings.enable_denoising,
            "normalization_enabled": settings.enable_normalization,
            "enhanced_settings": "Optimized for Indian accents"
        }
    
    def _create_speaker_labeled_text(self, whisper_result: Dict, speaker_segments: List[Dict]) -> str:
        """
        Create combined text with speaker labels included.
        
        Args:
            whisper_result: Whisper transcription result
            speaker_segments: List of speaker segments with timing
            
        Returns:
            String with speaker labels embedded in the text
        """
        if not speaker_segments or not whisper_result.get("segments"):
            return whisper_result.get("text", "")
        
        # Get Whisper segments with timing
        whisper_segments = whisper_result["segments"]
        
        # Create a timeline of speaker changes
        speaker_timeline = []
        for segment in speaker_segments:
            speaker_timeline.append({
                "time": segment["start"],
                "speaker": segment["speaker"],
                "type": "start"
            })
            speaker_timeline.append({
                "time": segment["end"],
                "speaker": segment["speaker"],
                "type": "end"
            })
        
        # Sort timeline by time
        speaker_timeline.sort(key=lambda x: x["time"])
        
        # Process each Whisper segment and add speaker labels
        labeled_segments = []
        current_speakers = set()
        
        for whisper_seg in whisper_segments:
            seg_start = whisper_seg["start"]
            seg_end = whisper_seg["end"]
            seg_text = whisper_seg["text"].strip()
            
            # Find which speakers are active during this segment
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
            
            # If no speakers detected, use the most recent speaker or default
            if not active_speakers:
                if current_speakers:
                    active_speakers = current_speakers
                else:
                    active_speakers = {"Speaker 1"}
            
            current_speakers = active_speakers
            
            # Create speaker label
            if len(active_speakers) == 1:
                speaker_label = f"[{list(active_speakers)[0]}]"
            else:
                # Multiple speakers - use the one with most overlap
                speaker_label = f"[{list(active_speakers)[0]}]"  # Use first speaker for simplicity
            
            # Add labeled segment
            if seg_text:
                labeled_segments.append(f"{speaker_label} {seg_text}")
        
        return " ".join(labeled_segments)

    def transcribe_audio(self, audio_path, model_size="base"):
        """Use faster model for quicker processing"""
        # Use base or small model instead of large
        model = whisper.load_model(model_size, device="cpu")  # or "cuda" if available
        result = model.transcribe(audio_path)
        return result


class TranscriptionService:
    """Service layer for transcription operations."""
    
    def __init__(self):
        self.engine = TranscriptionEngine(
            model_size=settings.whisper_model,
            device=settings.device
        )
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> Dict:
        """Transcribe a file using the transcription engine."""
        return self.engine.transcribe(file_path, language)
    
    def get_engine_info(self) -> Dict:
        """Get information about the transcription engine."""
        return self.engine.get_model_info()


# Global transcription engine instance
transcription_engine = TranscriptionEngine(
    model_size=settings.whisper_model,
    device=settings.device
) 