# Audio Processing Pipeline

## Overview

The audio processing pipeline is the foundational component that prepares audio data for transcription and speaker identification. It handles format conversion, quality enhancement, segmentation, and preprocessing to optimize downstream machine learning model performance.

## Pipeline Architecture

```
Audio Input → Validation → Preprocessing → Segmentation → Enhancement → Output
     ↓             ↓            ↓             ↓             ↓           ↓
File Upload   Format Check   Conversion   VAD/Silence   Noise Reduction  Chunks
Metadata      Quality Check  Normalization  Detection    Audio Enhancement Ready for ML
```

## Processing Stages

### 1. Audio File Validation

**Supported Formats**
- **Primary**: WAV, MP3, M4A, FLAC
- **Secondary**: AAC, OGG, WMA, AIFF
- **Professional**: BWF (Broadcast Wave Format), RF64

**Validation Criteria**
```python
class AudioValidation:
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_DURATION = 4 * 3600  # 4 hours
    MIN_DURATION = 1  # 1 second
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
    SUPPORTED_BIT_DEPTHS = [16, 24, 32]
    MAX_CHANNELS = 8
    
    def validate(self, audio_file):
        # File size validation
        if audio_file.size > self.MAX_FILE_SIZE:
            raise ValidationError("File too large")
        
        # Format validation
        if not self.is_supported_format(audio_file.format):
            raise ValidationError("Unsupported format")
        
        # Audio properties validation
        metadata = self.extract_metadata(audio_file)
        if metadata.duration > self.MAX_DURATION:
            raise ValidationError("Audio too long")
        
        return ValidationResult(valid=True, metadata=metadata)
```

**Metadata Extraction**
```python
import librosa
import mutagen

def extract_metadata(file_path):
    # Audio technical metadata
    audio_info = mutagen.File(file_path)
    
    # Load with librosa for analysis
    y, sr = librosa.load(file_path, sr=None)
    
    metadata = {
        'duration': len(y) / sr,
        'sample_rate': sr,
        'channels': 1 if y.ndim == 1 else y.shape[0],
        'bit_depth': audio_info.info.bits_per_sample if hasattr(audio_info.info, 'bits_per_sample') else None,
        'format': audio_info.mime[0] if audio_info.mime else None,
        'bitrate': audio_info.info.bitrate if hasattr(audio_info.info, 'bitrate') else None,
        'codec': audio_info.info.codec if hasattr(audio_info.info, 'codec') else None
    }
    
    return metadata
```

### 2. Audio Preprocessing

**Format Conversion**
```python
from pydub import AudioSegment

class AudioConverter:
    TARGET_SAMPLE_RATE = 16000  # Optimal for speech recognition
    TARGET_CHANNELS = 1         # Mono conversion
    TARGET_FORMAT = 'wav'       # Uncompressed for processing
    
    def convert(self, input_file):
        # Load audio with pydub
        audio = AudioSegment.from_file(input_file)
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample to target sample rate
        if audio.frame_rate != self.TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(self.TARGET_SAMPLE_RATE)
        
        # Normalize audio levels
        audio = self.normalize_audio(audio)
        
        return audio
    
    def normalize_audio(self, audio):
        # Peak normalization to prevent clipping
        current_peak = audio.max
        target_peak = -3.0  # dBFS
        
        if current_peak > target_peak:
            gain_reduction = target_peak - current_peak
            audio = audio + gain_reduction
        
        return audio
```

**Audio Quality Assessment**
```python
import numpy as np
from scipy import signal

class AudioQualityAnalyzer:
    def analyze_quality(self, audio_data, sample_rate):
        quality_metrics = {}
        
        # Signal-to-Noise Ratio (SNR)
        quality_metrics['snr'] = self.calculate_snr(audio_data)
        
        # Dynamic Range
        quality_metrics['dynamic_range'] = self.calculate_dynamic_range(audio_data)
        
        # Spectral Centroid (brightness)
        quality_metrics['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        ).mean()
        
        # Zero Crossing Rate (speech characteristic)
        quality_metrics['zcr'] = librosa.feature.zero_crossing_rate(audio_data).mean()
        
        # Overall quality score
        quality_metrics['quality_score'] = self.calculate_quality_score(quality_metrics)
        
        return quality_metrics
    
    def calculate_snr(self, audio_data):
        # Simple SNR calculation
        signal_power = np.var(audio_data)
        noise_power = np.var(audio_data - signal.medfilt(audio_data, kernel_size=5))
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
```

### 3. Voice Activity Detection (VAD)

**Advanced VAD Implementation**
```python
import webrtcvad
import librosa

class VoiceActivityDetector:
    def __init__(self, aggressiveness=2):
        self.vad = webrtcvad.Vad(aggressiveness)  # 0-3, higher = more aggressive
        self.frame_duration = 30  # ms
        self.sample_rate = 16000
    
    def detect_speech_segments(self, audio_data):
        # Convert to required format for WebRTC VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Frame-based VAD
        frame_size = int(self.sample_rate * self.frame_duration / 1000)
        frames = self.frame_generator(audio_int16, frame_size)
        
        speech_segments = []
        current_segment = None
        
        for i, frame in enumerate(frames):
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            timestamp = i * self.frame_duration / 1000.0
            
            if is_speech:
                if current_segment is None:
                    current_segment = {'start': timestamp, 'end': timestamp}
                else:
                    current_segment['end'] = timestamp
            else:
                if current_segment is not None:
                    # Add buffer around speech segments
                    current_segment['start'] = max(0, current_segment['start'] - 0.1)
                    current_segment['end'] += 0.1
                    speech_segments.append(current_segment)
                    current_segment = None
        
        # Handle case where audio ends with speech
        if current_segment is not None:
            speech_segments.append(current_segment)
        
        return speech_segments
    
    def frame_generator(self, audio, frame_size):
        """Generate frames of specified size from audio."""
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            yield audio[i:i + frame_size]

class AdvancedVAD:
    """More sophisticated VAD using energy and spectral features."""
    
    def __init__(self):
        self.energy_threshold = 0.01
        self.zcr_threshold = 0.1
        self.spectral_threshold = 1000
    
    def detect_speech_advanced(self, audio_data, sample_rate):
        # Energy-based detection
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.01 * sample_rate)     # 10ms hop
        
        # Extract features
        energy = librosa.feature.rms(
            y=audio_data, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        zcr = librosa.feature.zero_crossing_rate(
            audio_data, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=sample_rate,
            hop_length=hop_length
        )[0]
        
        # Combine features for speech detection
        speech_frames = (
            (energy > self.energy_threshold) &
            (zcr < self.zcr_threshold) &
            (spectral_centroid > self.spectral_threshold)
        )
        
        # Convert frame-based detection to time segments
        time_frames = librosa.frames_to_time(
            np.arange(len(speech_frames)), 
            sr=sample_rate, 
            hop_length=hop_length
        )
        
        return self.frames_to_segments(speech_frames, time_frames)
```

### 4. Audio Segmentation

**Intelligent Segmentation Strategy**
```python
class AudioSegmenter:
    def __init__(self):
        self.max_segment_length = 30  # seconds
        self.min_segment_length = 5   # seconds
        self.overlap_duration = 1     # seconds for context
        self.silence_threshold = 0.01
        self.min_silence_duration = 0.5  # seconds
    
    def segment_audio(self, audio_data, sample_rate, speech_segments=None):
        if speech_segments is None:
            vad = VoiceActivityDetector()
            speech_segments = vad.detect_speech_segments(audio_data)
        
        # Merge nearby speech segments
        merged_segments = self.merge_close_segments(speech_segments)
        
        # Split long segments
        final_segments = []
        for segment in merged_segments:
            if segment['end'] - segment['start'] > self.max_segment_length:
                split_segments = self.split_long_segment(
                    audio_data, sample_rate, segment
                )
                final_segments.extend(split_segments)
            else:
                final_segments.append(segment)
        
        return final_segments
    
    def merge_close_segments(self, segments, gap_threshold=2.0):
        """Merge speech segments that are close together."""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # If gap is small, merge segments
            if current['start'] - last['end'] < gap_threshold:
                last['end'] = current['end']
            else:
                merged.append(current)
        
        return merged
    
    def split_long_segment(self, audio_data, sample_rate, segment):
        """Split long segments at natural pause points."""
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]
        
        # Find pause points (low energy regions)
        pause_points = self.find_pause_points(segment_audio, sample_rate)
        
        # Create subsegments
        subsegments = []
        current_start = segment['start']
        
        for pause_time in pause_points:
            absolute_pause_time = segment['start'] + pause_time
            
            if absolute_pause_time - current_start >= self.min_segment_length:
                subsegments.append({
                    'start': current_start,
                    'end': absolute_pause_time
                })
                current_start = absolute_pause_time
        
        # Add final segment
        if segment['end'] - current_start >= self.min_segment_length:
            subsegments.append({
                'start': current_start,
                'end': segment['end']
            })
        
        return subsegments
    
    def find_pause_points(self, audio_data, sample_rate):
        """Find natural pause points in audio."""
        # Calculate energy in short windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = int(0.05 * sample_rate)    # 50ms hop
        
        energy = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            energy.append(np.mean(window ** 2))
        
        energy = np.array(energy)
        
        # Find low energy regions
        energy_threshold = np.percentile(energy, 25)  # Bottom quartile
        low_energy_frames = energy < energy_threshold
        
        # Find continuous low energy regions
        pause_candidates = []
        in_pause = False
        pause_start = 0
        
        for i, is_low_energy in enumerate(low_energy_frames):
            time = i * hop_size / sample_rate
            
            if is_low_energy and not in_pause:
                in_pause = True
                pause_start = time
            elif not is_low_energy and in_pause:
                in_pause = False
                pause_duration = time - pause_start
                
                if pause_duration >= self.min_silence_duration:
                    pause_candidates.append(pause_start + pause_duration / 2)
        
        return pause_candidates
```

### 5. Audio Enhancement

**Noise Reduction**
```python
import noisereduce as nr
from scipy.signal import butter, filtfilt

class AudioEnhancer:
    def __init__(self):
        self.noise_reduction_strength = 0.8
        self.enable_spectral_gating = True
        self.enable_bandpass_filter = True
        self.speech_band_low = 80   # Hz
        self.speech_band_high = 8000  # Hz
    
    def enhance_audio(self, audio_data, sample_rate):
        enhanced_audio = audio_data.copy()
        
        # 1. Noise reduction
        if self.noise_reduction_strength > 0:
            enhanced_audio = self.reduce_noise(enhanced_audio, sample_rate)
        
        # 2. Bandpass filtering for speech
        if self.enable_bandpass_filter:
            enhanced_audio = self.apply_bandpass_filter(
                enhanced_audio, sample_rate
            )
        
        # 3. Dynamic range compression
        enhanced_audio = self.apply_compression(enhanced_audio)
        
        # 4. Spectral gating for further noise reduction
        if self.enable_spectral_gating:
            enhanced_audio = self.spectral_gating(enhanced_audio, sample_rate)
        
        return enhanced_audio
    
    def reduce_noise(self, audio_data, sample_rate):
        """Reduce background noise using spectral subtraction."""
        try:
            # Use noisereduce library for basic noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_data, 
                sr=sample_rate,
                prop_decrease=self.noise_reduction_strength
            )
            return reduced_noise
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio_data
    
    def apply_bandpass_filter(self, audio_data, sample_rate):
        """Apply bandpass filter to focus on speech frequencies."""
        nyquist = sample_rate / 2
        low = self.speech_band_low / nyquist
        high = self.speech_band_high / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def apply_compression(self, audio_data, threshold=0.2, ratio=4.0):
        """Apply dynamic range compression."""
        # Simple compressor implementation
        compressed = audio_data.copy()
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression ratio
        compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
            threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
        )
        
        return compressed
    
    def spectral_gating(self, audio_data, sample_rate):
        """Advanced spectral gating for noise reduction."""
        # STFT parameters
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Spectral gating
        gate_threshold = np.percentile(magnitude, 30)  # Adaptive threshold
        
        # Create gate mask
        gate_mask = magnitude > gate_threshold
        
        # Apply smoothing to gate mask
        from scipy.ndimage import median_filter
        gate_mask = median_filter(gate_mask.astype(float), size=(3, 3)) > 0.5
        
        # Apply gate
        gated_magnitude = magnitude * gate_mask
        
        # Reconstruct audio
        gated_stft = gated_magnitude * np.exp(1j * phase)
        gated_audio = librosa.istft(gated_stft, hop_length=hop_length)
        
        return gated_audio
```

### 6. Performance Optimization

**Parallel Processing**
```python
import concurrent.futures
from functools import partial

class AudioProcessorOptimized:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.processor = AudioProcessor()
    
    def process_batch(self, audio_files):
        """Process multiple audio files in parallel."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all processing jobs
            future_to_file = {
                executor.submit(self.processor.process, audio_file): audio_file
                for audio_file in audio_files
            }
            
            results = {}
            for future in concurrent.futures.as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    results[audio_file] = result
                except Exception as exc:
                    print(f'Audio file {audio_file} generated an exception: {exc}')
                    results[audio_file] = None
        
        return results
    
    def process_segments_parallel(self, audio_segments):
        """Process audio segments in parallel."""
        process_func = partial(self.processor.enhance_audio)
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            enhanced_segments = list(executor.map(process_func, audio_segments))
        
        return enhanced_segments
```

**Memory Optimization**
```python
class MemoryEfficientProcessor:
    def __init__(self, chunk_size=1048576):  # 1MB chunks
        self.chunk_size = chunk_size
    
    def process_large_file(self, file_path):
        """Process large audio files in chunks to manage memory."""
        # Get file info without loading entire file
        with soundfile.SoundFile(file_path) as f:
            sample_rate = f.samplerate
            channels = f.channels
            frames = f.frames
        
        # Process in chunks
        processed_chunks = []
        frames_per_chunk = self.chunk_size // channels
        
        with soundfile.SoundFile(file_path) as f:
            while True:
                chunk = f.read(frames_per_chunk)
                if chunk.size == 0:
                    break
                
                # Process chunk
                processed_chunk = self.process_audio_chunk(chunk, sample_rate)
                processed_chunks.append(processed_chunk)
                
                # Clear memory
                del chunk
        
        # Combine processed chunks
        return np.concatenate(processed_chunks)
    
    def process_audio_chunk(self, chunk, sample_rate):
        """Process individual audio chunk."""
        # Apply processing with minimal memory footprint
        enhanced = self.lightweight_enhancement(chunk, sample_rate)
        return enhanced
```

## Configuration and Parameters

**Processing Configuration**
```yaml
# audio_processing_config.yaml
audio_processing:
  validation:
    max_file_size_mb: 500
    max_duration_hours: 4
    min_duration_seconds: 1
    supported_formats: ['wav', 'mp3', 'm4a', 'flac', 'aac']
  
  conversion:
    target_sample_rate: 16000
    target_channels: 1
    target_format: 'wav'
    normalization_peak_db: -3.0
  
  vad:
    aggressiveness: 2  # 0-3, WebRTC VAD
    frame_duration_ms: 30
    speech_buffer_ms: 100
  
  segmentation:
    max_segment_length_seconds: 30
    min_segment_length_seconds: 5
    overlap_duration_seconds: 1
    silence_threshold: 0.01
    min_silence_duration_seconds: 0.5
  
  enhancement:
    noise_reduction_strength: 0.8
    enable_spectral_gating: true
    enable_bandpass_filter: true
    speech_band_low_hz: 80
    speech_band_high_hz: 8000
    compression_threshold: 0.2
    compression_ratio: 4.0
  
  performance:
    max_workers: 4
    chunk_size_mb: 1
    enable_gpu_acceleration: true
    memory_limit_mb: 2048
```

**Quality Thresholds**
```python
class QualityConfig:
    # Minimum quality thresholds
    MIN_SNR_DB = 10
    MIN_DYNAMIC_RANGE_DB = 20
    MAX_NOISE_LEVEL = 0.1
    MIN_SPEECH_PERCENTAGE = 0.1  # 10% of audio should be speech
    
    # Warning thresholds
    WARNING_SNR_DB = 15
    WARNING_DYNAMIC_RANGE_DB = 30
    
    # Quality scoring weights
    WEIGHTS = {
        'snr': 0.3,
        'dynamic_range': 0.2,
        'spectral_centroid': 0.2,
        'speech_percentage': 0.3
    }
```

## Integration with ML Pipeline

**Preparation for Transcription Models**
```python
class MLReadyProcessor:
    def prepare_for_whisper(self, audio_data, sample_rate):
        """Prepare audio specifically for Whisper model."""
        # Whisper expects 16kHz mono
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, sample_rate, 16000)
        
        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data)
        
        # Pad or trim to 30 seconds (Whisper's context window)
        target_length = 16000 * 30  # 30 seconds at 16kHz
        if len(audio_data) > target_length:
            # Split into chunks
            chunks = []
            for i in range(0, len(audio_data), target_length):
                chunk = audio_data[i:i + target_length]
                if len(chunk) < target_length:
                    # Pad with zeros
                    chunk = np.pad(chunk, (0, target_length - len(chunk)))
                chunks.append(chunk)
            return chunks
        else:
            # Pad to target length
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            return [audio_data]
    
    def prepare_for_speaker_diarization(self, audio_data, sample_rate):
        """Prepare audio for pyannote.audio speaker diarization."""
        # pyannote typically works well with 16kHz
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, sample_rate, 16000)
        
        # Keep original length for diarization
        return audio_data, 16000
```

## Error Handling and Recovery

**Robust Error Handling**
```python
class ProcessingError(Exception):
    def __init__(self, message, error_type, recoverable=False):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable

class RobustAudioProcessor:
    def __init__(self):
        self.retry_attempts = 3
        self.fallback_processors = [
            'basic_processor',
            'minimal_processor'
        ]
    
    def process_with_recovery(self, audio_file):
        """Process with automatic error recovery."""
        for attempt in range(self.retry_attempts):
            try:
                return self.process_audio(audio_file)
            except ProcessingError as e:
                if e.recoverable and attempt < self.retry_attempts - 1:
                    print(f"Retrying processing (attempt {attempt + 1}): {e}")
                    continue
                elif e.error_type == 'format_error':
                    return self.try_fallback_processing(audio_file)
                else:
                    raise
            except Exception as e:
                print(f"Unexpected error in processing: {e}")
                if attempt < self.retry_attempts - 1:
                    continue
                else:
                    return self.minimal_processing(audio_file)
    
    def try_fallback_processing(self, audio_file):
        """Try alternative processing methods."""
        for processor_name in self.fallback_processors:
            try:
                processor = getattr(self, processor_name)
                return processor(audio_file)
            except Exception as e:
                print(f"Fallback processor {processor_name} failed: {e}")
                continue
        
        raise ProcessingError("All processing methods failed", "critical", False)
```

This comprehensive audio processing pipeline provides the foundation for high-quality transcription and speaker identification while maintaining robustness and performance.