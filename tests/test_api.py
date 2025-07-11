import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "transcription-platform"
    
    def test_web_interface(self, client):
        """Test web interface endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_upload_file(self, client, temp_upload_dir, sample_audio_files):
        """Test file upload endpoint."""
        with open(sample_audio_files[0], "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "filename" in data
    
    def test_upload_invalid_file(self, client):
        """Test upload with invalid file."""
        files = {"file": ("test.txt", b"not audio", "text/plain")}
        response = client.post("/api/upload", files=files)
        assert response.status_code == 400
    
    def test_list_files(self, client):
        """Test list files endpoint."""
        response = client.get("/api/files")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_file_info(self, client):
        """Test get file info endpoint."""
        # First upload a file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF    WAVEfmt ")  # Minimal WAV header
        
        try:
            with open(f.name, "rb") as audio_file:
                files = {"file": ("test.wav", audio_file, "audio/wav")}
                upload_response = client.post("/api/upload", files=files)
            
            if upload_response.status_code == 200:
                file_id = upload_response.json()["file_id"]
                
                # Get file info
                response = client.get(f"/api/files/{file_id}")
                assert response.status_code == 200
                data = response.json()
                assert "filename" in data
                assert "size" in data
        finally:
            os.unlink(f.name)
    
    def test_delete_file(self, client):
        """Test delete file endpoint."""
        # First upload a file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF    WAVEfmt ")
        
        try:
            with open(f.name, "rb") as audio_file:
                files = {"file": ("test.wav", audio_file, "audio/wav")}
                upload_response = client.post("/api/upload", files=files)
            
            if upload_response.status_code == 200:
                file_id = upload_response.json()["file_id"]
                
                # Delete file
                response = client.delete(f"/api/files/{file_id}")
                assert response.status_code == 200
                
                # Verify file is deleted
                get_response = client.get(f"/api/files/{file_id}")
                assert get_response.status_code == 404
        finally:
            os.unlink(f.name)
    
    def test_transcribe_file(self, client):
        """Test transcribe file endpoint."""
        # First upload a file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF    WAVEfmt ")
        
        try:
            with open(f.name, "rb") as audio_file:
                files = {"file": ("test.wav", audio_file, "audio/wav")}
                upload_response = client.post("/api/upload", files=files)
            
            if upload_response.status_code == 200:
                file_id = upload_response.json()["file_id"]
                
                # Start transcription
                with patch('src.transcription.engine.TranscriptionEngine.transcribe') as mock_transcribe:
                    mock_transcribe.return_value = {
                        "text": "Test transcription",
                        "language": "en",
                        "confidence": 0.9,
                        "segments": [],
                        "error": None
                    }
                    
                    response = client.post(f"/api/transcribe", json={
                        "file_id": file_id,
                        "language": "en"
                    })
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "job_id" in data
        finally:
            os.unlink(f.name)
    
    def test_get_job_status(self, client):
        """Test get job status endpoint."""
        response = client.get("/api/jobs/1")
        # Should return 404 for non-existent job
        assert response.status_code == 404
    
    def test_list_jobs(self, client):
        """Test list jobs endpoint."""
        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_download_transcription(self, client):
        """Test download transcription endpoint."""
        response = client.get("/api/transcriptions/1/download")
        # Should return 404 for non-existent transcription
        assert response.status_code == 404
    
    def test_get_transcription(self, client):
        """Test get transcription endpoint."""
        response = client.get("/api/transcriptions/1")
        # Should return 404 for non-existent transcription
        assert response.status_code == 404


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post("/api/upload")
        assert response.status_code == 422  # Validation error
    
    def test_upload_large_file(self, client):
        """Test upload with file too large."""
        # Create a large file (simulate)
        large_content = b"x" * (100 * 1024 * 1024 + 1)  # 100MB + 1 byte
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(large_content)
        
        try:
            with open(f.name, "rb") as audio_file:
                files = {"file": ("large.wav", audio_file, "audio/wav")}
                response = client.post("/api/upload", files=files)
            
            # Should fail due to file size
            assert response.status_code in [400, 413]
        finally:
            os.unlink(f.name)
    
    def test_invalid_file_id(self, client):
        """Test operations with invalid file ID."""
        response = client.get("/api/files/999999")
        assert response.status_code == 404
        
        response = client.delete("/api/files/999999")
        assert response.status_code == 404
    
    def test_invalid_job_id(self, client):
        """Test operations with invalid job ID."""
        response = client.get("/api/jobs/999999")
        assert response.status_code == 404
    
    def test_invalid_transcription_id(self, client):
        """Test operations with invalid transcription ID."""
        response = client.get("/api/transcriptions/999999")
        assert response.status_code == 404 