# API Usage Guide

This guide covers the REST API endpoints for the transcription platform.

## Base URL

```
http://127.0.0.1:8000/api
```

## Authentication

Currently, the API does not require authentication for local development.

## Endpoints

### Health Check

**GET** `/health`

Check if the service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "transcription-platform"
}
```

### File Upload

**POST** `/upload`

Upload an audio file for transcription.

**Form Data:**
- `file`: Audio file (WAV, MP3, M4A, FLAC)
- `language` (optional): Language code (en, es, fr, etc.)

**Response:**
```json
{
  "file_id": 1,
  "filename": "audio.wav",
  "status": "uploaded",
  "size": 1024000,
  "duration": 30.5
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@audio.wav" \
  -F "language=en"
```

### List Files

**GET** `/files`

Get list of uploaded audio files.

**Query Parameters:**
- `limit` (optional): Number of files to return (default: 50)
- `offset` (optional): Number of files to skip (default: 0)

**Response:**
```json
[
  {
    "id": 1,
    "filename": "audio.wav",
    "size": 1024000,
    "duration": 30.5,
    "format": "wav",
    "status": "uploaded",
    "created_at": "2024-01-01T12:00:00Z"
  }
]
```

### Get File Info

**GET** `/files/{file_id}`

Get detailed information about a specific file.

**Response:**
```json
{
  "id": 1,
  "filename": "audio.wav",
  "original_filename": "my_audio.wav",
  "size": 1024000,
  "duration": 30.5,
  "format": "wav",
  "sample_rate": 16000,
  "channels": 1,
  "status": "uploaded",
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Delete File

**DELETE** `/files/{file_id}`

Delete an uploaded file and its associated data.

**Response:**
```json
{
  "message": "File deleted successfully"
}
```

### Start Transcription

**POST** `/transcribe`

Start transcription for an uploaded file.

**Request Body:**
```json
{
  "file_id": 1,
  "language": "en",
  "enable_speaker_diarization": true
}
```

**Response:**
```json
{
  "job_id": 1,
  "status": "pending",
  "message": "Transcription job started"
}
```

### Get Job Status

**GET** `/jobs/{job_id}`

Get the status of a transcription job.

**Response:**
```json
{
  "id": 1,
  "status": "completed",
  "progress": 100.0,
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:01:30Z",
  "error_message": null
}
```

### List Jobs

**GET** `/jobs`

Get list of transcription jobs.

**Query Parameters:**
- `limit` (optional): Number of jobs to return (default: 50)
- `offset` (optional): Number of jobs to skip (default: 0)
- `status` (optional): Filter by status (pending, processing, completed, failed)

**Response:**
```json
[
  {
    "id": 1,
    "status": "completed",
    "progress": 100.0,
    "created_at": "2024-01-01T12:00:00Z",
    "audio_file": {
      "id": 1,
      "filename": "audio.wav"
    }
  }
]
```

### Get Transcription

**GET** `/transcriptions/{job_id}`

Get transcription results for a completed job.

**Response:**
```json
{
  "id": 1,
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "confidence": 0.95,
  "processing_time": 30.5,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is",
      "speaker": "SPEAKER_00",
      "confidence": 0.95
    },
    {
      "start": 2.5,
      "end": 4.0,
      "text": "a test transcription.",
      "speaker": "SPEAKER_00",
      "confidence": 0.92
    }
  ],
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "total_duration": 4.0,
      "segment_count": 2
    }
  ],
  "created_at": "2024-01-01T12:01:30Z"
}
```

### Download Transcription

**GET** `/transcriptions/{job_id}/download`

Download transcription as a text file.

**Response:** Text file with transcription content.

**Headers:**
```
Content-Type: text/plain
Content-Disposition: attachment; filename="transcription_1.txt"
```

## Error Responses

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (resource doesn't exist)
- `422`: Validation Error (invalid file format, etc.)
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, there are no rate limits for local development.

## File Size Limits

- Maximum file size: 100MB
- Supported formats: WAV, MP3, M4A, FLAC
- Recommended audio quality: 16kHz, mono

## WebSocket Support (Future)

Real-time progress updates will be available via WebSocket:

```
ws://127.0.0.1:8000/ws/jobs/{job_id}
```

## Examples

### Complete Workflow

```bash
# 1. Upload file
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@audio.wav" \
  -F "language=en"

# Response: {"file_id": 1, "status": "uploaded"}

# 2. Start transcription
curl -X POST http://127.0.0.1:8000/api/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_id": 1, "enable_speaker_diarization": true}'

# Response: {"job_id": 1, "status": "pending"}

# 3. Check status
curl http://127.0.0.1:8000/api/jobs/1

# 4. Get results
curl http://127.0.0.1:8000/api/transcriptions/1

# 5. Download results
curl http://127.0.0.1:8000/api/transcriptions/1/download > transcription.txt
```

### Python Client Example

```python
import requests

# Upload file
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/api/upload',
        files={'file': f},
        data={'language': 'en'}
    )
    file_data = response.json()

# Start transcription
response = requests.post(
    'http://127.0.0.1:8000/api/transcribe',
    json={
        'file_id': file_data['file_id'],
        'enable_speaker_diarization': True
    }
)
job_data = response.json()

# Poll for completion
import time
while True:
    response = requests.get(f"http://127.0.0.1:8000/api/jobs/{job_data['job_id']}")
    job_status = response.json()
    
    if job_status['status'] == 'completed':
        break
    elif job_status['status'] == 'failed':
        raise Exception(f"Job failed: {job_status['error_message']}")
    
    time.sleep(2)

# Get results
response = requests.get(f"http://127.0.0.1:8000/api/transcriptions/{job_data['job_id']}")
transcription = response.json()
print(transcription['text'])
```

## Testing

Test the API using the included test suite:

```bash
pytest tests/test_api.py -v
```

Or use the web interface at: http://127.0.0.1:8000 