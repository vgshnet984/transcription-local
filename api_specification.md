# API Specification

## Overview

The Transcription Platform provides a comprehensive REST API for audio transcription with speaker identification. The API follows RESTful principles and provides both synchronous and asynchronous processing capabilities.

## Base URL

```
Production: https://api.transcription-platform.com/v1
Staging: https://staging-api.transcription-platform.com/v1
Development: http://localhost:8000/api/v1
```

## Authentication

All API endpoints require authentication using JWT tokens.

### Authentication Flow

```http
POST /auth/login
Content-Type: application/json

{
    "email": "user@example.com",
    "password": "password123"
}
```

**Response:**
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

### Using Authentication

Include the JWT token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Core Endpoints

### 1. File Management

#### Upload Audio File

```http
POST /files/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

Form Data:
- file: <audio_file>
- metadata: <optional_json_metadata>
```

**Response:**
```json
{
    "file_id": "uuid-string",
    "filename": "meeting_recording.wav",
    "size": 52428800,
    "duration": 1800.5,
    "format": "wav",
    "sample_rate": 16000,
    "channels": 1,
    "status": "uploaded",
    "upload_time": "2024-01-01T12:00:00Z",
    "metadata": {
        "source": "zoom_meeting",
        "meeting_id": "123456789"
    }
}
```

#### Get File Information

```http
GET /files/{file_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
    "file_id": "uuid-string",
    "filename": "meeting_recording.wav",
    "size": 52428800,
    "duration": 1800.5,
    "format": "wav",
    "sample_rate": 16000,
    "channels": 1,
    "status": "processed",
    "upload_time": "2024-01-01T12:00:00Z",
    "processing_time": "2024-01-01T12:01:30Z",
    "metadata": {},
    "quality_metrics": {
        "snr_db": 25.3,
        "dynamic_range_db": 45.2,
        "quality_score": 0.85
    }
}
```

#### List Files

```http
GET /files?page=1&limit=50&status=processed&format=wav
Authorization: Bearer <token>
```

**Response:**
```json
{
    "files": [
        {
            "file_id": "uuid-string",
            "filename": "meeting_recording.wav",
            "size": 52428800,
            "duration": 1800.5,
            "status": "processed",
            "upload_time": "2024-01-01T12:00:00Z"
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 50,
        "total": 150,
        "total_pages": 3
    }
}
```

#### Delete File

```http
DELETE /files/{file_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
    "message": "File deleted successfully",
    "file_id": "uuid-string"
}
```

### 2. Transcription Jobs

#### Create Transcription Job

```http
POST /transcription/jobs
Content-Type: application/json
Authorization: Bearer <token>

{
    "file_id": "uuid-string",
    "config": {
        "model": "whisper-base",
        "language": "auto",
        "enable_speaker_diarization": true,
        "enable_speaker_identification": true,
        "max_speakers": 10,
        "include_timestamps": true,
        "include_confidence_scores": true,
        "custom_vocabulary": ["API", "machine learning", "transcription"],
        "output_formats": ["json", "srt", "vtt"]
    },
    "webhook_url": "https://example.com/webhook",
    "priority": "normal"
}
```

**Response:**
```json
{
    "job_id": "uuid-string",
    "file_id": "uuid-string",
    "status": "queued",
    "config": {
        "model": "whisper-base",
        "language": "auto",
        "enable_speaker_diarization": true,
        "enable_speaker_identification": true,
        "max_speakers": 10,
        "include_timestamps": true,
        "include_confidence_scores": true,
        "custom_vocabulary": ["API", "machine learning", "transcription"],
        "output_formats": ["json", "srt", "vtt"]
    },
    "created_at": "2024-01-01T12:00:00Z",
    "estimated_completion": "2024-01-01T12:15:00Z",
    "webhook_url": "https://example.com/webhook",
    "priority": "normal"
}
```

#### Get Job Status

```http
GET /transcription/jobs/{job_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
    "job_id": "uuid-string",
    "file_id": "uuid-string",
    "status": "processing",
    "progress": 65.5,
    "config": {
        "model": "whisper-base",
        "language": "en",
        "enable_speaker_diarization": true,
        "enable_speaker_identification": true
    },
    "created_at": "2024-01-01T12:00:00Z",
    "started_at": "2024-01-01T12:02:00Z",
    "estimated_completion": "2024-01-01T12:15:00Z",
    "processing_stages": {
        "audio_preprocessing": "completed",
        "speech_to_text": "processing",
        "speaker_diarization": "queued",
        "speaker_identification": "queued",
        "post_processing": "queued"
    },
    "current_stage": "speech_to_text",
    "error": null
}
```

#### List Jobs

```http
GET /transcription/jobs?page=1&limit=20&status=completed&user_id=current
Authorization: Bearer <token>
```

**Response:**
```json
{
    "jobs": [
        {
            "job_id": "uuid-string",
            "file_id": "uuid-string",
            "status": "completed",
            "progress": 100,
            "created_at": "2024-01-01T12:00:00Z",
            "completed_at": "2024-01-01T12:14:32Z",
            "processing_time_seconds": 874
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 20,
        "total": 45,
        "total_pages": 3
    }
}
```

#### Cancel Job

```http
DELETE /transcription/jobs/{job_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
    "message": "Job cancelled successfully",
    "job_id": "uuid-string",
    "status": "cancelled"
}
```

### 3. Transcription Results

#### Get Transcription

```http
GET /transcription/{transcription_id}?format=json&include_speakers=true
Authorization: Bearer <token>
```

**Response:**
```json
{
    "transcription_id": "uuid-string",
    "job_id": "uuid-string",
    "file_id": "uuid-string",
    "language": "en",
    "confidence": 0.92,
    "processing_time_seconds": 874,
    "word_count": 2543,
    "speaker_count": 3,
    "duration": 1800.5,
    "segments": [
        {
            "id": 0,
            "start_time": 0.0,
            "end_time": 5.2,
            "text": "Welcome everyone to today's meeting.",
            "confidence": 0.95,
            "speaker_id": "SPEAKER_00",
            "speaker_name": "John Smith",
            "speaker_confidence": 0.89,
            "words": [
                {
                    "word": "Welcome",
                    "start_time": 0.0,
                    "end_time": 0.5,
                    "confidence": 0.98
                },
                {
                    "word": "everyone",
                    "start_time": 0.6,
                    "end_time": 1.1,
                    "confidence": 0.94
                }
            ]
        }
    ],
    "speakers": {
        "SPEAKER_00": {
            "name": "John Smith",
            "speaking_time": 450.2,
            "word_count": 892,
            "identified": true,
            "confidence": 0.89
        },
        "SPEAKER_01": {
            "name": "Unknown Speaker 1",
            "speaking_time": 680.1,
            "word_count": 1245,
            "identified": false,
            "confidence": 0.75
        },
        "SPEAKER_02": {
            "name": "Sarah Johnson",
            "speaking_time": 670.2,
            "word_count": 406,
            "identified": true,
            "confidence": 0.92
        }
    },
    "metadata": {
        "model_used": "whisper-base",
        "processing_date": "2024-01-01T12:14:32Z",
        "audio_quality": {
            "snr_db": 25.3,
            "quality_score": 0.85
        }
    }
}
```

#### Get Transcription in Different Formats

**SRT Format:**
```http
GET /transcription/{transcription_id}?format=srt
Authorization: Bearer <token>
Content-Type: text/plain
```

**Response:**
```srt
1
00:00:00,000 --> 00:00:05,200
[John Smith] Welcome everyone to today's meeting.

2
00:00:05,500 --> 00:00:12,300
[John Smith] We have several important topics to discuss today.

3
00:00:12,600 --> 00:00:18,400
[Sarah Johnson] Thank you John. I'd like to start with the quarterly results.
```

**VTT Format:**
```http
GET /transcription/{transcription_id}?format=vtt
Authorization: Bearer <token>
Content-Type: text/vtt
```

**Response:**
```vtt
WEBVTT

00:00:00.000 --> 00:00:05.200
<v John Smith>Welcome everyone to today's meeting.

00:00:05.500 --> 00:00:12.300
<v John Smith>We have several important topics to discuss today.

00:00:12.600 --> 00:00:18.400
<v Sarah Johnson>Thank you John. I'd like to start with the quarterly results.
```

**Plain Text Format:**
```http
GET /transcription/{transcription_id}?format=txt&include_speakers=true
Authorization: Bearer <token>
Content-Type: text/plain
```

**Response:**
```text
[John Smith]: Welcome everyone to today's meeting. We have several important topics to discuss today.

[Sarah Johnson]: Thank you John. I'd like to start with the quarterly results.

[Unknown Speaker 1]: That sounds good. Do we have the numbers ready?
```

#### Search Transcriptions

```http
GET /transcription/search?q=quarterly+results&user_id=current&date_from=2024-01-01&date_to=2024-01-31
Authorization: Bearer <token>
```

**Response:**
```json
{
    "results": [
        {
            "transcription_id": "uuid-string",
            "file_id": "uuid-string",
            "filename": "meeting_recording.wav",
            "matches": [
                {
                    "segment_id": 15,
                    "start_time": 125.6,
                    "end_time": 130.2,
                    "text": "I'd like to start with the quarterly results",
                    "speaker_name": "Sarah Johnson",
                    "relevance_score": 0.95
                }
            ],
            "created_at": "2024-01-15T10:30:00Z"
        }
    ],
    "total_results": 1,
    "query": "quarterly results",
    "search_time_ms": 45
}
```

### 4. Speaker Management

#### Create Speaker Profile

```http
POST /speakers/profiles
Content-Type: application/json
Authorization: Bearer <token>

{
    "name": "John Smith",
    "organization_id": "uuid-string",
    "voice_samples": [
        {
            "file_id": "uuid-string",
            "start_time": 10.0,
            "end_time": 20.0,
            "quality_score": 0.9
        }
    ],
    "metadata": {
        "title": "CEO",
        "department": "Executive"
    }
}
```

**Response:**
```json
{
    "profile_id": "uuid-string",
    "name": "John Smith",
    "organization_id": "uuid-string",
    "voice_samples_count": 1,
    "training_quality": 0.85,
    "created_at": "2024-01-01T12:00:00Z",
    "metadata": {
        "title": "CEO",
        "department": "Executive"
    },
    "status": "active"
}
```

#### Get Speaker Profile

```http
GET /speakers/profiles/{profile_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
    "profile_id": "uuid-string",
    "name": "John Smith",
    "organization_id": "uuid-string",
    "voice_samples": [
        {
            "sample_id": "uuid-string",
            "file_id": "uuid-string",
            "start_time": 10.0,
            "end_time": 20.0,
            "quality_score": 0.9,
            "added_at": "2024-01-01T12:00:00Z"
        }
    ],
    "training_quality": 0.85,
    "identification_accuracy": 0.92,
    "total_identifications": 45,
    "created_at": "2024-01-01T12:00:00Z",
    "last_updated": "2024-01-15T14:30:00Z",
    "metadata": {
        "title": "CEO",
        "department": "Executive"
    },
    "status": "active"
}
```

#### List Speaker Profiles

```http
GET /speakers/profiles?organization_id=uuid-string&status=active&page=1&limit=20
Authorization: Bearer <token>
```

**Response:**
```json
{
    "profiles": [
        {
            "profile_id": "uuid-string",
            "name": "John Smith",
            "training_quality": 0.85,
            "identification_accuracy": 0.92,
            "total_identifications": 45,
            "status": "active",
            "created_at": "2024-01-01T12:00:00Z"
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 20,
        "total": 12,
        "total_pages": 1
    }
}
```

#### Update Speaker Profile

```http
PUT /speakers/profiles/{profile_id}
Content-Type: application/json
Authorization: Bearer <token>

{
    "name": "John A. Smith",
    "metadata": {
        "title": "Chief Executive Officer",
        "department": "Executive"
    },
    "status": "active"
}
```

#### Add Voice Sample

```http
POST /speakers/profiles/{profile_id}/voice-samples
Content-Type: application/json
Authorization: Bearer <token>

{
    "file_id": "uuid-string",
    "start_time": 45.0,
    "end_time": 55.0,
    "quality_score": 0.88
}
```

#### Identify Speaker

```http
POST /speakers/identify
Content-Type: application/json
Authorization: Bearer <token>

{
    "file_id": "uuid-string",
    "start_time": 10.0,
    "end_time": 20.0,
    "organization_id": "uuid-string"
}
```

**Response:**
```json
{
    "identification": {
        "profile_id": "uuid-string",
        "speaker_name": "John Smith",
        "confidence": 0.89,
        "similarity_score": 0.92
    },
    "alternatives": [
        {
            "profile_id": "uuid-string-2",
            "speaker_name": "Mike Johnson",
            "confidence": 0.65,
            "similarity_score": 0.78
        }
    ],
    "processing_time_ms": 150
}
```

### 5. Real-time Transcription

#### WebSocket Connection

```javascript
// WebSocket connection for real-time transcription
const ws = new WebSocket('wss://api.transcription-platform.com/v1/ws/transcribe');

// Authentication
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your-jwt-token'
    }));
};

// Configure transcription
ws.send(JSON.stringify({
    type: 'config',
    config: {
        language: 'en',
        enable_speaker_diarization: true,
        sample_rate: 16000,
        format: 'raw'
    }
}));

// Send audio data
ws.send(audioBuffer); // Binary audio data

// Receive transcription results
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Transcription:', data);
};
```

**Real-time Response Format:**
```json
{
    "type": "transcription",
    "segment_id": 123,
    "start_time": 45.2,
    "end_time": 48.7,
    "text": "This is a real-time transcription",
    "confidence": 0.87,
    "is_final": false,
    "speaker_id": "SPEAKER_00",
    "speaker_name": "John Smith",
    "speaker_confidence": 0.82
}
```

### 6. Analytics and Statistics

#### Get User Statistics

```http
GET /analytics/user/stats?period=30d
Authorization: Bearer <token>
```

**Response:**
```json
{
    "period": "30d",
    "total_files": 156,
    "total_transcription_time": 45000.5,
    "total_processing_time": 12500.2,
    "average_accuracy": 0.91,
    "total_speakers_identified": 234,
    "files_by_format": {
        "wav": 89,
        "mp3": 45,
        "m4a": 22
    },
    "processing_by_model": {
        "whisper-base": 120,
        "whisper-small": 36
    },
    "quality_distribution": {
        "high": 123,
        "medium": 28,
        "low": 5
    }
}
```

#### Get Organization Analytics

```http
GET /analytics/organization/{org_id}/stats?period=90d
Authorization: Bearer <token>
```

**Response:**
```json
{
    "organization_id": "uuid-string",
    "period": "90d",
    "users": 25,
    "total_files": 1250,
    "total_transcription_time": 356000.8,
    "total_processing_time": 89500.3,
    "cost_breakdown": {
        "transcription": 2450.50,
        "storage": 125.30,
        "api_calls": 89.20
    },
    "accuracy_metrics": {
        "average": 0.89,
        "best": 0.98,
        "worst": 0.72
    },
    "popular_features": [
        "speaker_diarization",
        "real_time_transcription",
        "custom_vocabulary"
    ]
}
```

## Error Handling

### Standard Error Response Format

```json
{
    "error": {
        "code": "INVALID_FILE_FORMAT",
        "message": "The uploaded file format is not supported",
        "details": {
            "supported_formats": ["wav", "mp3", "m4a", "flac"],
            "received_format": "avi"
        },
        "timestamp": "2024-01-01T12:00:00Z",
        "request_id": "uuid-string"
    }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid authentication token |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `INVALID_FILE_FORMAT` | 400 | Unsupported audio file format |
| `FILE_TOO_LARGE` | 413 | File exceeds maximum size limit |
| `QUOTA_EXCEEDED` | 429 | User has exceeded their usage quota |
| `PROCESSING_FAILED` | 500 | Transcription processing failed |
| `MODEL_UNAVAILABLE` | 503 | Requested transcription model is unavailable |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests in time window |

## Rate Limiting

### Rate Limit Headers

All API responses include rate limiting information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint Category | Limit | Window |
|------------------|-------|--------|
| File Upload | 100 requests | 1 hour |
| Transcription Jobs | 500 requests | 1 hour |
| Real-time WebSocket | 10 connections | Per user |
| Speaker Management | 200 requests | 1 hour |
| Analytics | 100 requests | 1 hour |

## Webhooks

### Webhook Configuration

Configure webhooks to receive notifications about job completion:

```http
POST /webhooks
Content-Type: application/json
Authorization: Bearer <token>

{
    "url": "https://example.com/webhook",
    "events": ["job.completed", "job.failed", "speaker.identified"],
    "secret": "webhook-secret-key",
    "active": true
}
```

### Webhook Payload

```json
{
    "event": "job.completed",
    "timestamp": "2024-01-01T12:15:00Z",
    "data": {
        "job_id": "uuid-string",
        "file_id": "uuid-string",
        "transcription_id": "uuid-string",
        "status": "completed",
        "processing_time_seconds": 874,
        "accuracy": 0.92,
        "speaker_count": 3
    },
    "signature": "sha256=..."
}
```

### Webhook Security

Verify webhook authenticity using HMAC signature:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(
        f"sha256={expected_signature}",
        signature
    )
```

## SDKs and Code Examples

### Python SDK

```python
from transcription_client import TranscriptionClient

# Initialize client
client = TranscriptionClient(
    api_key="your-api-key",
    base_url="https://api.transcription-platform.com/v1"
)

# Upload and transcribe
with open("meeting.wav", "rb") as audio_file:
    # Upload file
    file_info = client.upload_file(audio_file)
    
    # Create transcription job
    job = client.create_transcription_job(
        file_id=file_info["file_id"],
        config={
            "enable_speaker_diarization": True,
            "enable_speaker_identification": True,
            "language": "en"
        }
    )
    
    # Wait for completion
    result = client.wait_for_completion(job["job_id"])
    
    # Get transcription
    transcription = client.get_transcription(result["transcription_id"])
    
    print(f"Transcription completed with {len(transcription['segments'])} segments")
    print(f"Identified {len(transcription['speakers'])} speakers")
```

### JavaScript SDK

```javascript
import { TranscriptionClient } from '@transcription-platform/js-sdk';

const client = new TranscriptionClient({
    apiKey: 'your-api-key',
    baseURL: 'https://api.transcription-platform.com/v1'
});

// Upload and transcribe
async function transcribeAudio(audioFile) {
    try {
        // Upload file
        const fileInfo = await client.uploadFile(audioFile);
        
        // Create transcription job
        const job = await client.createTranscriptionJob({
            fileId: fileInfo.file_id,
            config: {
                enableSpeakerDiarization: true,
                enableSpeakerIdentification: true,
                language: 'auto'
            }
        });
        
        // Poll for completion
        const result = await client.pollJobCompletion(job.job_id);
        
        // Get transcription
        const transcription = await client.getTranscription(result.transcription_id);
        
        return transcription;
    } catch (error) {
        console.error('Transcription failed:', error);
        throw error;
    }
}
```

### cURL Examples

**Upload and transcribe in one request:**
```bash
curl -X POST "https://api.transcription-platform.com/v1/transcription/direct" \
     -H "Authorization: Bearer your-jwt-token" \
     -F "file=@meeting.wav" \
     -F "config={\"enable_speaker_diarization\":true,\"language\":\"en\"}"
```

**Check job status:**
```bash
curl -X GET "https://api.transcription-platform.com/v1/transcription/jobs/uuid-string" \
     -H "Authorization: Bearer your-jwt-token"
```

**Download transcription as SRT:**
```bash
curl -X GET "https://api.transcription-platform.com/v1/transcription/uuid-string?format=srt" \
     -H "Authorization: Bearer your-jwt-token" \
     -o transcription.srt
```

## Pagination

All list endpoints support pagination using the following parameters:

- `page`: Page number (1-based, default: 1)
- `limit`: Items per page (max: 100, default: 20)
- `sort`: Sort field (default: created_at)
- `order`: Sort order (asc/desc, default: desc)

**Example:**
```http
GET /transcription/jobs?page=2&limit=50&sort=created_at&order=asc
```

## Filtering and Sorting

### Common Filter Parameters

- `status`: Filter by status (queued, processing, completed, failed, cancelled)
- `date_from`: Filter by creation date (ISO 8601 format)
- `date_to`: Filter by creation date (ISO 8601 format)
- `user_id`: Filter by user (use "current" for authenticated user)
- `organization_id`: Filter by organization

### Examples

```http
# Get completed jobs from last week
GET /transcription/jobs?status=completed&date_from=2024-01-01&date_to=2024-01-07

# Get files uploaded today
GET /files?date_from=2024-01-01T00:00:00Z&date_to=2024-01-01T23:59:59Z

# Search transcriptions by content
GET /transcription/search?q=machine+learning&date_from=2024-01-01
```

This API specification provides comprehensive documentation for integrating with the transcription platform, covering all major functionality including file management, transcription processing, speaker identification, and real-time capabilities.