import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.transcription.engine import TranscriptionEngine


class TestTranscriptionEngine:
    """Test cases for the transcription engine."""
    
    def test_init(self):
        """Test transcription engine initialization."""
        engine = TranscriptionEngine()
        assert engine.model_size == "tiny"
        assert engine.model is not None  # Model loaded in __init__
    
    def test_load_model(self):
        """Test model loading."""
        with patch('whisper.load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            engine = TranscriptionEngine()
            model = engine._load_model()
            
            assert model == mock_model
            mock_load.assert_called_once_with("base")
    
    def test_transcribe_basic(self, sample_audio_files):
        """Test basic transcription functionality."""
        with patch('whisper.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.return_value = {
                "text": "Hello world",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello world",
                        "no_speech_prob": 0.1
                    }
                ],
                "language": "en"
            }
            mock_load.return_value = mock_model
            
            engine = TranscriptionEngine()
            result = engine.transcribe(sample_audio_files[0])
            
            assert result["text"] == "Hello world"
            assert result["language"] == "en"
            assert result["error"] is None
            assert len(result["segments"]) == 1
    
    def test_transcribe_with_speaker_diarization(self, sample_audio_files):
        """Test transcription with speaker diarization."""
        with patch('whisper.load_model') as mock_load, \
             patch('src.config.settings.enable_speaker_diarization', True), \
             patch('src.transcription.engine.DIARIZATION_AVAILABLE', True):
            
            # Mock Whisper model
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "text": "Hello world",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello world",
                        "avg_logprob": -0.1
                    }
                ],
                "language": "en"
            }
            mock_load.return_value = mock_model
            
            # Mock diarization pipeline
            with patch('pyannote.audio.Pipeline.from_pretrained') as mock_pipeline:
                mock_diarization = MagicMock()
                mock_diarization.itertracks.return_value = [
                    (MagicMock(start=0.0, end=2.0), None, "SPEAKER_00")
                ]
                mock_pipeline.return_value = mock_diarization
                
                engine = TranscriptionEngine()
                result = engine.transcribe(sample_audio_files[0])
                
                assert result["text"] == "Hello world"
                assert len(result["speaker_segments"]) > 0
                assert result["speakers"][0]["speaker_id"] == "SPEAKER_00"
    
    def test_transcribe_error_handling(self, sample_audio_files):
        """Test transcription error handling."""
        with patch('whisper.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.side_effect = Exception("Model error")
            mock_load.return_value = mock_model
            
            engine = TranscriptionEngine()
            result = engine.transcribe(sample_audio_files[0])
            
            assert result["error"] is not None
            assert "Model error" in result["error"]
            assert result["text"] == ""
    
    def test_transcribe_invalid_file(self):
        """Test transcription with invalid file."""
        engine = TranscriptionEngine()
        result = engine.transcribe("nonexistent_file.wav")
        
        assert result["error"] is not None
        assert result["text"] == ""
    
    def test_transcribe_with_language(self, sample_audio_files):
        """Test transcription with specific language."""
        with patch('whisper.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.return_value = {
                "text": "Bonjour le monde",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Bonjour le monde",
                        "no_speech_prob": 0.1
                    }
                ],
                "language": "fr"
            }
            mock_load.return_value = mock_model
            
            engine = TranscriptionEngine()
            result = engine.transcribe(sample_audio_files[0], language="fr")
            
            assert result["text"] == "Bonjour le monde"
            assert result["language"] == "fr"
    
    def test_get_model_info(self):
        """Test getting model information."""
        engine = TranscriptionEngine()
        info = engine.get_model_info()

        assert "model_size" in info
        assert "available_sizes" in info
        assert info["model_size"] == "tiny"
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        engine = TranscriptionEngine()
        
        result = {
            "segments": [
                {"avg_logprob": -0.1},
                {"avg_logprob": -0.2}
            ]
        }
        
        confidence = engine._calculate_confidence(result)
        assert 0.0 <= confidence <= 1.0
    
    def test_extract_speaker_info(self):
        """Test speaker information extraction."""
        engine = TranscriptionEngine()
        
        speaker_segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "duration": 2.0
            },
            {
                "start": 2.0,
                "end": 4.0,
                "speaker": "SPEAKER_00",
                "duration": 2.0
            }
        ]
        
        speakers = engine._extract_speaker_info(speaker_segments)
        
        assert len(speakers) == 1
        assert speakers[0]["speaker_id"] == "SPEAKER_00"
        assert speakers[0]["total_duration"] == 4.0
        assert speakers[0]["segment_count"] == 2 