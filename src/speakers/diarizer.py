import os
from typing import List, Dict, Any

# Lazy import to avoid Windows multiprocessing issues
import platform
if platform.system() != 'Windows':
    try:
        from pyannote.audio import Pipeline
        DIARIZATION_AVAILABLE = True
    except ImportError:
        DIARIZATION_AVAILABLE = False
else:
    DIARIZATION_AVAILABLE = False

class Diarizer:
    def __init__(self, hf_token=None):
        self.available = DIARIZATION_AVAILABLE
        self.pipeline = None
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if self.available and self.hf_token:
            try:
                # Lazy import for Windows compatibility
                if platform.system() == 'Windows':
                    try:
                        from pyannote.audio import Pipeline
                    except ImportError:
                        self.available = False
                        self.pipeline = None
                        self.error = "pyannote.audio not available on Windows"
                        return
                else:
                    from pyannote.audio import Pipeline
                
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=self.hf_token
                )
            except Exception as e:
                self.available = False
                self.pipeline = None
                self.error = str(e)
        elif self.available:
            self.available = False
            self.error = "HF_TOKEN not set"
        else:
            self.error = "pyannote.audio not installed"

    def diarize(self, audio_path: str, max_speakers: int = 10) -> List[Dict[str, Any]]:
        if self.available and self.pipeline:
            try:
                result = self.pipeline(audio_path, num_speakers=max_speakers)
                segments = []
                for turn, _, speaker in result.itertracks(yield_label=True):
                    segments.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": speaker,
                        "confidence": 1.0  # pyannote does not provide per-segment confidence
                    })
                return segments
            except Exception as e:
                return self._single_speaker_fallback(audio_path, error=str(e))
        else:
            return self._single_speaker_fallback(audio_path, error=getattr(self, 'error', None))

    def _single_speaker_fallback(self, audio_path, error=None):
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return [{
            "start": 0.0,
            "end": duration,
            "speaker": "Speaker 1",
            "confidence": 1.0,
            "error": error or "Diarization unavailable"
        }] 