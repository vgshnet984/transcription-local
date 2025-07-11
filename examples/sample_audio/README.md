# Sample Audio Files

This directory contains sample audio files for testing the transcription platform.

## File Descriptions

### Single Speaker Samples
- `single_speaker_short.wav` - 5-10 second single speaker recording
- `single_speaker_medium.wav` - 30-60 second single speaker recording
- `single_speaker_long.wav` - 2-5 minute single speaker recording

### Multi-Speaker Samples
- `conversation_2speakers.wav` - 2-person conversation (1-2 minutes)
- `conversation_3speakers.wav` - 3-person conversation (2-3 minutes)
- `meeting_sample.wav` - Meeting recording with multiple speakers (5-10 minutes)

### Different Formats
- `sample_mp3.mp3` - MP3 format sample
- `sample_m4a.m4a` - M4A format sample
- `sample_flac.flac` - FLAC format sample

### Quality Levels
- `high_quality.wav` - High quality recording (44.1kHz, 16-bit)
- `medium_quality.wav` - Medium quality recording (22.05kHz, 16-bit)
- `low_quality.wav` - Low quality recording (8kHz, 8-bit)

### Special Cases
- `noise_sample.wav` - Audio with background noise
- `music_sample.wav` - Audio with background music
- `silence_sample.wav` - Audio with long silence periods

## Usage

These files are used for:
1. Testing transcription accuracy
2. Validating speaker diarization
3. Performance benchmarking
4. Format conversion testing
5. Quality assessment

## Generation

Sample files can be generated using:
- Text-to-speech tools
- Recording equipment
- Audio editing software
- Online audio generators

## Notes

- All files should be properly licensed for testing
- File sizes should be reasonable for testing (under 50MB each)
- Include a variety of accents, languages, and speaking styles
- Ensure privacy compliance (no real personal data)

## Getting Sample Audio

You can add your own audio files here for testing, or download sample files from:

- **LibriVox**: Free public domain audiobooks (https://librivox.org/)
- **Common Voice**: Mozilla's open-source voice dataset (https://commonvoice.mozilla.org/)
- **Audio samples**: Various free audio samples online

## Supported Formats

The platform supports these audio formats:
- WAV
- MP3  
- M4A
- FLAC

## Testing

To test with a sample file:

```bash
# Test transcription
python scripts/test_transcription.py examples/sample_audio/your_file.wav

# Test with specific language
python scripts/test_transcription.py examples/sample_audio/your_file.wav --language en
```

## File Size Limits

- Maximum file size: 100MB (configurable)
- Recommended: 1-10MB for quick testing
- Longer files will take more time to process

## Tips

- Use clear speech with minimal background noise for best results
- English audio works best with the default 'tiny' model
- For other languages, specify the language code when testing 