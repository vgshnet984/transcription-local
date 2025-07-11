import pytest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
from src.audio.processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for audio processing."""
    
    def test_init(self):
        """Test audio processor initialization."""
        processor = AudioProcessor()
        assert processor.supported_formats is not None
        assert processor.max_file_size > 0
        assert processor.sample_rate > 0
    
    def test_validate(self, sample_audio_files):
        """Test audio file validation."""
        processor = AudioProcessor()
        
        # Valid file
        result = processor.validate(sample_audio_files[0])
        assert result["valid"] is True
        
        # Invalid file
        result = processor.validate("nonexistent.wav")
        assert result["valid"] is False
        assert "File not found" in result["error"]
    
    def test_validate_large_file(self):
        """Test validation of large file."""
        processor = AudioProcessor()
        
        # Create a file larger than max size
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * (processor.max_file_size + 1))
            large_file = f.name
        
        try:
            result = processor.validate(large_file)
            assert result["valid"] is False
            assert "File too large" in result["error"]
        finally:
            os.unlink(large_file)
    
    def test_validate_invalid_format(self):
        """Test validation of invalid format."""
        processor = AudioProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not audio")
            invalid_file = f.name
        
        try:
            result = processor.validate(invalid_file)
            assert result["valid"] is False
            assert "Unsupported format" in result["error"]
        finally:
            os.unlink(invalid_file)
    
    def test_save_uploaded_file(self, temp_upload_dir):
        """Test saving uploaded file."""
        processor = AudioProcessor()
        
        # Create a mock uploaded file
        file_content = b"RIFF    WAVEfmt "  # Minimal WAV header
        filename = "test_audio.wav"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Mock the uploaded file
            mock_file = MagicMock()
            mock_file.filename = filename
            mock_file.file = open(temp_file_path, 'rb')
            
            file_path, file_info = processor.save_uploaded_file(mock_file, filename)
            
            assert os.path.exists(file_path)
            assert file_info["filename"] == filename
            assert file_info["file_size"] == len(file_content)
            assert file_info["format"] == "wav"
            
            mock_file.file.close()
        finally:
            os.unlink(temp_file_path)
    
    def test_get_metadata(self, sample_audio_files):
        """Test audio metadata extraction."""
        processor = AudioProcessor()
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.get_duration') as mock_duration, \
             patch('pydub.AudioSegment.from_file') as mock_audio_segment:
            
            mock_load.return_value = (np.zeros(16000), 16000)  # 1 second of silence
            mock_duration.return_value = 1.0
            
            # Mock audio segment
            mock_segment = MagicMock()
            mock_segment.channels = 1
            mock_audio_segment.return_value = mock_segment
            
            metadata = processor.get_metadata(sample_audio_files[0])
            
            assert metadata["duration"] == 1.0
            assert metadata["sample_rate"] == 16000
            assert metadata["channels"] == 1
    
    def test_convert_to_wav(self, sample_audio_files):
        """Test audio format conversion."""
        processor = AudioProcessor()
        
        with patch('pydub.AudioSegment.from_file') as mock_from_file, \
             patch('tempfile.NamedTemporaryFile') as mock_temp:
            
            # Mock audio segment
            mock_audio = MagicMock()
            mock_audio.set_channels.return_value = mock_audio
            mock_audio.set_frame_rate.return_value = mock_audio
            mock_audio.export.return_value = None
            mock_from_file.return_value = mock_audio
            
            # Mock temp file
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.wav"
            mock_temp.return_value = mock_temp_file
            
            output_path = processor.convert_to_wav(sample_audio_files[0])
            
            assert output_path == "/tmp/test.wav"
            mock_from_file.assert_called_once()
            mock_audio.export.assert_called_once()
    
    def test_normalize_audio(self, sample_audio_files):
        """Test audio normalization."""
        processor = AudioProcessor()
        
        with patch('librosa.load') as mock_load, \
             patch('librosa.resample') as mock_resample, \
             patch('soundfile.write') as mock_write:
            
            mock_load.return_value = (np.zeros(16000), 8000)  # 8kHz audio
            mock_resample.return_value = np.zeros(16000)  # Resampled to 16kHz
            mock_write.return_value = None
            
            output_path = processor.normalize_audio(sample_audio_files[0])
            
            assert output_path.endswith(".wav")
            mock_load.assert_called_once()
            mock_resample.assert_called_once()
            mock_write.assert_called_once()
    
    def test_cleanup_file(self, temp_upload_dir):
        """Test file cleanup."""
        processor = AudioProcessor()
        
        # Create a test file
        test_file = os.path.join(temp_upload_dir, "test_file.wav")
        with open(test_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write("test content")
        
        assert os.path.exists(test_file)
        
        # Clean up file
        processor.cleanup_file(test_file)
        
        assert not os.path.exists(test_file)
    
    def test_supported_formats(self):
        """Test supported formats property."""
        processor = AudioProcessor()
        formats = processor.supported_formats
        
        assert isinstance(formats, list)
        assert "wav" in formats
        assert "mp3" in formats
    
    def test_error_handling(self):
        """Test error handling in audio processing."""
        processor = AudioProcessor()
        
        # Test with non-existent file
        result = processor.validate("nonexistent.wav")
        assert result["valid"] is False
        assert "File not found" in result["error"]
        
        # Test with invalid file format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not audio")
            invalid_file = f.name
        
        try:
            result = processor.validate(invalid_file)
            assert result["valid"] is False
            assert "Unsupported format" in result["error"]
        finally:
            os.unlink(invalid_file) 