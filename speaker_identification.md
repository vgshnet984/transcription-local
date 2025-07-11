# Speaker Identification & Diarization

## Overview

The speaker identification system combines speaker diarization (who spoke when) with speaker identification (who is this person) to provide comprehensive speaker-aware transcription. The system uses pyannote.audio for core diarization capabilities and implements custom identification algorithms for speaker recognition and tracking.

## Core Components

### 1. Speaker Diarization Pipeline

```
Audio Input → Segmentation → Feature Extraction → Clustering → Speaker Timeline
     ↓             ↓              ↓               ↓             ↓
   Preprocessing  VAD           Embeddings    Speaker IDs   Time-aligned
   Enhancement   Speech/Non    (512-dim)     Assignment    Speaker Labels
                 Speech        vectors                     + Confidence
```

**pyannote.audio Integration**
```python
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch

class SpeakerDiarizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load pre-trained diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None  # Add HuggingFace token if needed
        )
        
        # Load speaker embedding model
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )
        
        # Configuration
        self.min_speakers = 1
        self.max_speakers = 20
        self.min_segment_duration = 0.5  # seconds
        self.embedding_batch_size = 32
    
    def diarize(self, audio_file, num_speakers=None):
        """Perform speaker diarization on audio file."""
        try:
            # Run diarization pipeline
            diarization = self.diarization_pipeline(
                audio_file,
                num_speakers=num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Convert to speaker timeline
            speaker_timeline = self.convert_to_timeline(diarization)
            
            return speaker_timeline
            
        except Exception as e:
            print(f"Diarization failed: {e}")
            return self.fallback_diarization(audio_file)
    
    def convert_to_timeline(self, diarization):
        """Convert pyannote diarization to our timeline format."""
        timeline = []
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                'start_time': segment.start,
                'end_time': segment.end,
                'speaker_id': speaker,
                'duration': segment.end - segment.start,
                'confidence': 1.0  # pyannote doesn't provide confidence by default
            })
        
        return sorted(timeline, key=lambda x: x['start_time'])
    
    def fallback_diarization(self, audio_file):
        """Simple fallback diarization using clustering."""
        return self.simple_clustering_diarization(audio_file)
```

### 2. Speaker Embedding Generation

**Advanced Embedding System**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

class SpeakerEmbeddingSystem:
    def __init__(self):
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb"
        )
        self.embedding_dim = 512
        self.similarity_threshold = 0.8
        self.min_embedding_duration = 1.0  # seconds
    
    def extract_embeddings(self, audio_segments):
        """Extract speaker embeddings from audio segments."""
        embeddings = []
        
        for segment in audio_segments:
            if segment['duration'] < self.min_embedding_duration:
                continue
            
            # Extract audio for this segment
            audio_data = self.extract_segment_audio(segment)
            
            # Generate embedding
            embedding = self.embedding_model(audio_data)
            
            embeddings.append({
                'segment_id': segment['id'],
                'embedding': embedding.cpu().numpy(),
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['duration']
            })
        
        return embeddings
    
    def cluster_speakers(self, embeddings, distance_threshold=0.7):
        """Cluster embeddings to identify unique speakers."""
        if not embeddings:
            return {}
        
        # Stack embeddings for clustering
        embedding_matrix = np.stack([emb['embedding'] for emb in embeddings])
        
        # Use Agglomerative Clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embedding_matrix)
        
        # Map segments to speakers
        speaker_mapping = {}
        for embedding, cluster_id in zip(embeddings, cluster_labels):
            speaker_id = f"SPEAKER_{cluster_id:02d}"
            speaker_mapping[embedding['segment_id']] = {
                'speaker_id': speaker_id,
                'cluster_id': cluster_id,
                'embedding': embedding['embedding'],
                'confidence': self.calculate_cluster_confidence(
                    embedding['embedding'], 
                    embedding_matrix[cluster_labels == cluster_id]
                )
            }
        
        return speaker_mapping
    
    def calculate_cluster_confidence(self, embedding, cluster_embeddings):
        """Calculate confidence of embedding assignment to cluster."""
        if len(cluster_embeddings) == 1:
            return 1.0
        
        # Calculate mean similarity to other embeddings in cluster
        similarities = cosine_similarity([embedding], cluster_embeddings)[0]
        mean_similarity = np.mean(similarities[similarities < 1.0])  # Exclude self
        
        return float(mean_similarity)
```

### 3. Speaker Identification and Recognition

**Speaker Profile Management**
```python
class SpeakerProfileManager:
    def __init__(self, database_manager):
        self.db = database_manager
        self.similarity_threshold = 0.85
        self.min_training_samples = 3
        self.max_profile_embeddings = 10
    
    def create_speaker_profile(self, speaker_name, voice_samples):
        """Create a new speaker profile from voice samples."""
        # Extract embeddings from voice samples
        embeddings = []
        for sample in voice_samples:
            embedding = self.embedding_model(sample['audio_data'])
            embeddings.append(embedding.cpu().numpy())
        
        if len(embeddings) < self.min_training_samples:
            raise ValueError(f"Need at least {self.min_training_samples} voice samples")
        
        # Create speaker profile
        profile = {
            'name': speaker_name,
            'embeddings': embeddings,
            'mean_embedding': np.mean(embeddings, axis=0),
            'std_embedding': np.std(embeddings, axis=0),
            'created_at': datetime.utcnow(),
            'sample_count': len(embeddings)
        }
        
        # Store in database
        profile_id = self.db.create_speaker_profile(profile)
        
        return profile_id
    
    def identify_speaker(self, unknown_embedding):
        """Identify speaker from embedding against known profiles."""
        # Get all speaker profiles
        profiles = self.db.get_all_speaker_profiles()
        
        if not profiles:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for profile in profiles:
            # Calculate similarity to profile
            similarity = self.calculate_profile_similarity(
                unknown_embedding, 
                profile
            )
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = {
                    'profile_id': profile['id'],
                    'speaker_name': profile['name'],
                    'similarity': similarity,
                    'confidence': self.similarity_to_confidence(similarity)
                }
        
        return best_match
    
    def calculate_profile_similarity(self, embedding, profile):
        """Calculate similarity between embedding and speaker profile."""
        # Method 1: Similarity to mean embedding
        mean_similarity = cosine_similarity(
            [embedding], 
            [profile['mean_embedding']]
        )[0][0]
        
        # Method 2: Maximum similarity to any profile embedding
        max_similarity = 0.0
        for profile_embedding in profile['embeddings']:
            similarity = cosine_similarity([embedding], [profile_embedding])[0][0]
            max_similarity = max(max_similarity, similarity)
        
        # Method 3: Probabilistic approach using distribution
        prob_similarity = self.calculate_probabilistic_similarity(
            embedding, profile
        )
        
        # Combine methods
        final_similarity = (
            0.4 * mean_similarity +
            0.4 * max_similarity +
            0.2 * prob_similarity
        )
        
        return final_similarity
    
    def update_speaker_profile(self, profile_id, new_embedding):
        """Update existing speaker profile with new embedding."""
        profile = self.db.get_speaker_profile(profile_id)
        
        # Add new embedding
        profile['embeddings'].append(new_embedding)
        
        # Limit number of stored embeddings
        if len(profile['embeddings']) > self.max_profile_embeddings:
            # Remove oldest or least representative embeddings
            profile['embeddings'] = self.select_representative_embeddings(
                profile['embeddings']
            )
        
        # Recalculate statistics
        profile['mean_embedding'] = np.mean(profile['embeddings'], axis=0)
        profile['std_embedding'] = np.std(profile['embeddings'], axis=0)
        profile['sample_count'] = len(profile['embeddings'])
        
        # Update in database
        self.db.update_speaker_profile(profile_id, profile)
```

### 4. Speaker-Aware Transcription Integration

**Combined Processing**
```python
class SpeakerAwareTranscriber:
    def __init__(self, transcription_engine, speaker_diarizer):
        self.transcriber = transcription_engine
        self.diarizer = speaker_diarizer
        self.speaker_manager = SpeakerProfileManager()
        self.confidence_threshold = 0.7
        self.overlap_tolerance = 0.1  # seconds
    
    def transcribe_with_speakers(self, audio_file, enable_identification=True):
        """Perform transcription with speaker diarization and identification."""
        
        # Step 1: Get speaker timeline
        speaker_timeline = self.diarizer.diarize(audio_file)
        
        # Step 2: Extract speaker embeddings
        embeddings = self.extract_speaker_embeddings(audio_file, speaker_timeline)
        
        # Step 3: Identify speakers (if enabled and profiles exist)
        identified_speakers = {}
        if enable_identification:
            identified_speakers = self.identify_all_speakers(embeddings)
        
        # Step 4: Perform transcription on audio segments
        transcription_segments = self.transcriber.transcribe_segments(
            audio_file, speaker_timeline
        )
        
        # Step 5: Combine transcription with speaker information
        final_result = self.combine_transcription_and_speakers(
            transcription_segments, speaker_timeline, identified_speakers
        )
        
        return final_result
    
    def extract_speaker_embeddings(self, audio_file, speaker_timeline):
        """Extract embeddings for each speaker segment."""
        embeddings = {}
        
        for segment in speaker_timeline:
            speaker_id = segment['speaker_id']
            
            # Extract audio for this segment
            segment_audio = self.extract_audio_segment(
                audio_file, segment['start_time'], segment['end_time']
            )
            
            # Generate embedding
            embedding = self.diarizer.embedding_model(segment_audio)
            
            if speaker_id not in embeddings:
                embeddings[speaker_id] = []
            
            embeddings[speaker_id].append({
                'embedding': embedding.cpu().numpy(),
                'segment': segment,
                'confidence': segment.get('confidence', 1.0)
            })
        
        return embeddings
    
    def identify_all_speakers(self, speaker_embeddings):
        """Identify all speakers using stored profiles."""
        identified_speakers = {}
        
        for speaker_id, embedding_list in speaker_embeddings.items():
            # Use the highest confidence embedding for identification
            best_embedding = max(embedding_list, key=lambda x: x['confidence'])
            
            # Attempt identification
            identification = self.speaker_manager.identify_speaker(
                best_embedding['embedding']
            )
            
            if identification and identification['confidence'] > self.confidence_threshold:
                identified_speakers[speaker_id] = identification
            else:
                # Unknown speaker
                identified_speakers[speaker_id] = {
                    'speaker_name': f"Unknown Speaker {speaker_id}",
                    'confidence': 0.0,
                    'is_unknown': True
                }
        
        return identified_speakers
    
    def combine_transcription_and_speakers(self, transcription_segments, 
                                         speaker_timeline, identified_speakers):
        """Combine transcription results with speaker information."""
        
        combined_segments = []
        
        for trans_segment in transcription_segments:
            # Find overlapping speaker segments
            overlapping_speakers = self.find_overlapping_speakers(
                trans_segment, speaker_timeline
            )
            
            # Determine primary speaker for this transcription segment
            primary_speaker = self.determine_primary_speaker(
                trans_segment, overlapping_speakers
            )
            
            # Create combined segment
            combined_segment = {
                'start_time': trans_segment['start_time'],
                'end_time': trans_segment['end_time'],
                'text': trans_segment['text'],
                'confidence': trans_segment['confidence'],
                'speaker_id': primary_speaker['speaker_id'] if primary_speaker else None,
                'speaker_name': self.get_speaker_name(
                    primary_speaker, identified_speakers
                ) if primary_speaker else "Unknown",
                'speaker_confidence': primary_speaker['confidence'] if primary_speaker else 0.0,
                'overlapping_speakers': [
                    {
                        'speaker_id': spk['speaker_id'],
                        'speaker_name': self.get_speaker_name(spk, identified_speakers),
                        'overlap_ratio': spk['overlap_ratio']
                    }
                    for spk in overlapping_speakers
                ]
            }
            
            combined_segments.append(combined_segment)
        
        return {
            'segments': combined_segments,
            'speakers': identified_speakers,
            'statistics': self.calculate_speaker_statistics(combined_segments)
        }
    
    def find_overlapping_speakers(self, transcription_segment, speaker_timeline):
        """Find speaker segments that overlap with transcription segment."""
        overlapping = []
        
        trans_start = transcription_segment['start_time']
        trans_end = transcription_segment['end_time']
        
        for speaker_segment in speaker_timeline:
            spk_start = speaker_segment['start_time']
            spk_end = speaker_segment['end_time']
            
            # Calculate overlap
            overlap_start = max(trans_start, spk_start)
            overlap_end = min(trans_end, spk_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                trans_duration = trans_end - trans_start
                overlap_ratio = overlap_duration / trans_duration
                
                if overlap_ratio > self.overlap_tolerance:
                    overlapping.append({
                        **speaker_segment,
                        'overlap_ratio': overlap_ratio,
                        'overlap_duration': overlap_duration
                    })
        
        return sorted(overlapping, key=lambda x: x['overlap_ratio'], reverse=True)
    
    def determine_primary_speaker(self, transcription_segment, overlapping_speakers):
        """Determine the primary speaker for a transcription segment."""
        if not overlapping_speakers:
            return None
        
        # Primary speaker is the one with highest overlap ratio
        return overlapping_speakers[0]
    
    def get_speaker_name(self, speaker_segment, identified_speakers):
        """Get the display name for a speaker."""
        speaker_id = speaker_segment['speaker_id']
        
        if speaker_id in identified_speakers:
            return identified_speakers[speaker_id]['speaker_name']
        else:
            return f"Speaker {speaker_id}"
```

### 5. Advanced Speaker Features

**Speaker Change Detection**
```python
class SpeakerChangeDetector:
    def __init__(self):
        self.window_size = 1.0  # seconds
        self.step_size = 0.1    # seconds
        self.threshold = 0.3    # change detection threshold
        
    def detect_speaker_changes(self, audio_data, sample_rate):
        """Detect speaker change points in audio."""
        change_points = []
        
        # Extract embeddings in sliding windows
        window_samples = int(self.window_size * sample_rate)
        step_samples = int(self.step_size * sample_rate)
        
        embeddings = []
        timestamps = []
        
        for i in range(0, len(audio_data) - window_samples, step_samples):
            window = audio_data[i:i + window_samples]
            embedding = self.extract_embedding(window, sample_rate)
            embeddings.append(embedding)
            timestamps.append(i / sample_rate)
        
        # Detect changes using embedding similarity
        for i in range(1, len(embeddings)):
            similarity = cosine_similarity(
                [embeddings[i-1]], [embeddings[i]]
            )[0][0]
            
            if similarity < (1 - self.threshold):
                change_points.append({
                    'timestamp': timestamps[i],
                    'confidence': 1 - similarity,
                    'type': 'speaker_change'
                })
        
        return change_points

class OverlappingSpeechHandler:
    def __init__(self):
        self.overlap_threshold = 0.5  # seconds
        self.energy_ratio_threshold = 0.3
    
    def detect_overlapping_speech(self, audio_data, sample_rate, speaker_timeline):
        """Detect and handle overlapping speech segments."""
        overlapping_segments = []
        
        # Find potential overlaps in speaker timeline
        for i in range(len(speaker_timeline) - 1):
            current = speaker_timeline[i]
            next_segment = speaker_timeline[i + 1]
            
            # Check for temporal overlap
            if current['end_time'] > next_segment['start_time']:
                overlap_duration = current['end_time'] - next_segment['start_time']
                
                if overlap_duration > self.overlap_threshold:
                    overlapping_segments.append({
                        'start_time': next_segment['start_time'],
                        'end_time': current['end_time'],
                        'duration': overlap_duration,
                        'speakers': [current['speaker_id'], next_segment['speaker_id']],
                        'type': 'overlap'
                    })
        
        # Analyze energy distribution in overlapping segments
        for overlap in overlapping_segments:
            self.analyze_overlap_energy(audio_data, sample_rate, overlap)
        
        return overlapping_segments
    
    def analyze_overlap_energy(self, audio_data, sample_rate, overlap_segment):
        """Analyze energy distribution in overlapping speech."""
        start_sample = int(overlap_segment['start_time'] * sample_rate)
        end_sample = int(overlap_segment['end_time'] * sample_rate)
        overlap_audio = audio_data[start_sample:end_sample]
        
        # Spectral analysis to separate speakers
        n_fft = 2048
        stft = librosa.stft(overlap_audio, n_fft=n_fft)
        magnitude = np.abs(stft)
        
        # Simple energy-based separation (can be enhanced with source separation)
        total_energy = np.sum(magnitude)
        
        # Estimate energy distribution between speakers
        # This is a simplified approach - in practice, use more sophisticated
        # source separation techniques
        overlap_segment['energy_analysis'] = {
            'total_energy': float(total_energy),
            'dominant_frequencies': self.find_dominant_frequencies(magnitude),
            'separation_confidence': 0.5  # Placeholder
        }
```

**Voice Activity Detection for Speakers**
```python
class SpeakerSpecificVAD:
    def __init__(self):
        self.speaker_models = {}
        self.frame_length = 0.025  # 25ms
        self.frame_shift = 0.01    # 10ms
    
    def train_speaker_vad(self, speaker_id, training_audio):
        """Train speaker-specific VAD model."""
        # Extract speaker-specific features
        features = self.extract_speaker_features(training_audio)
        
        # Train a simple classifier (can be enhanced with ML models)
        speaker_profile = {
            'mean_energy': np.mean(features['energy']),
            'std_energy': np.std(features['energy']),
            'mean_zcr': np.mean(features['zcr']),
            'std_zcr': np.std(features['zcr']),
            'spectral_profile': np.mean(features['mfcc'], axis=0)
        }
        
        self.speaker_models[speaker_id] = speaker_profile
    
    def detect_speaker_activity(self, audio_data, sample_rate, speaker_id):
        """Detect voice activity for specific speaker."""
        if speaker_id not in self.speaker_models:
            # Fall back to general VAD
            return self.general_vad(audio_data, sample_rate)
        
        model = self.speaker_models[speaker_id]
        features = self.extract_speaker_features(audio_data)
        
        # Calculate likelihood of speaker activity
        activity_scores = []
        
        for i in range(len(features['energy'])):
            # Energy score
            energy_score = self.calculate_feature_score(
                features['energy'][i], 
                model['mean_energy'], 
                model['std_energy']
            )
            
            # ZCR score
            zcr_score = self.calculate_feature_score(
                features['zcr'][i],
                model['mean_zcr'],
                model['std_zcr']
            )
            
            # Spectral score
            spectral_score = cosine_similarity(
                [features['mfcc'][i]], 
                [model['spectral_profile']]
            )[0][0]
            
            # Combined score
            combined_score = (energy_score + zcr_score + spectral_score) / 3
            activity_scores.append(combined_score)
        
        # Convert scores to binary decisions
        threshold = 0.6
        activity_frames = np.array(activity_scores) > threshold
        
        # Convert frame-based decisions to time segments
        return self.frames_to_segments(activity_frames, sample_rate)
```

### 6. Quality Assessment and Metrics

**Speaker Identification Metrics**
```python
class SpeakerMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_diarization_error_rate(self, predicted_timeline, ground_truth_timeline):
        """Calculate Diarization Error Rate (DER)."""
        # DER = (False Alarm + Missed Detection + Speaker Error) / Total
        
        total_time = max(
            max(seg['end_time'] for seg in predicted_timeline),
            max(seg['end_time'] for seg in ground_truth_timeline)
        )
        
        # Create time-aligned comparison
        time_resolution = 0.01  # 10ms resolution
        num_frames = int(total_time / time_resolution)
        
        predicted_labels = np.full(num_frames, -1)
        ground_truth_labels = np.full(num_frames, -1)
        
        # Fill predicted labels
        for segment in predicted_timeline:
            start_frame = int(segment['start_time'] / time_resolution)
            end_frame = int(segment['end_time'] / time_resolution)
            speaker_idx = hash(segment['speaker_id']) % 1000  # Simple mapping
            predicted_labels[start_frame:end_frame] = speaker_idx
        
        # Fill ground truth labels
        for segment in ground_truth_timeline:
            start_frame = int(segment['start_time'] / time_resolution)
            end_frame = int(segment['end_time'] / time_resolution)
            speaker_idx = hash(segment['speaker_id']) % 1000
            ground_truth_labels[start_frame:end_frame] = speaker_idx
        
        # Calculate errors
        speech_frames = ground_truth_labels >= 0
        total_speech_time = np.sum(speech_frames) * time_resolution
        
        if total_speech_time == 0:
            return 0.0
        
        # Missed detection: speech in ground truth but not in predicted
        missed_detection = np.sum(
            (ground_truth_labels >= 0) & (predicted_labels < 0)
        ) * time_resolution
        
        # False alarm: speech in predicted but not in ground truth
        false_alarm = np.sum(
            (predicted_labels >= 0) & (ground_truth_labels < 0)
        ) * time_resolution
        
        # Speaker error: both have speech but different speakers
        speaker_error = np.sum(
            (ground_truth_labels >= 0) & 
            (predicted_labels >= 0) & 
            (ground_truth_labels != predicted_labels)
        ) * time_resolution
        
        der = (missed_detection + false_alarm + speaker_error) / total_speech_time
        
        return {
            'der': der,
            'missed_detection_rate': missed_detection / total_speech_time,
            'false_alarm_rate': false_alarm / total_speech_time,
            'speaker_error_rate': speaker_error / total_speech_time
        }
    
    def calculate_identification_accuracy(self, predictions, ground_truth):
        """Calculate speaker identification accuracy."""
        if not predictions or not ground_truth:
            return 0.0
        
        correct_identifications = 0
        total_identifications = len(predictions)
        
        for pred_speaker, true_speaker in zip(predictions, ground_truth):
            if pred_speaker == true_speaker:
                correct_identifications += 1
        
        accuracy = correct_identifications / total_identifications
        
        return {
            'accuracy': accuracy,
            'correct_identifications': correct_identifications,
            'total_identifications': total_identifications
        }
```

### 7. Configuration and Optimization

**Speaker System Configuration**
```yaml
# speaker_config.yaml
speaker_diarization:
  model:
    diarization_pipeline: "pyannote/speaker-diarization-3.1"
    embedding_model: "speechbrain/spkrec-ecapa-voxceleb"
    device: "auto"  # auto, cpu, cuda
  
  parameters:
    min_speakers: 1
    max_speakers: 20
    min_segment_duration: 0.5
    embedding_batch_size: 32
    similarity_threshold: 0.8
  
  clustering:
    algorithm: "agglomerative"
    distance_threshold: 0.7
    linkage: "average"
    metric: "cosine"
  
  identification:
    similarity_threshold: 0.85
    confidence_threshold: 0.7
    min_training_samples: 3
    max_profile_embeddings: 10
  
  quality:
    overlap_tolerance: 0.1
    energy_ratio_threshold: 0.3
    change_detection_threshold: 0.3
  
  performance:
    enable_gpu: true
    batch_processing: true
    parallel_embedding_extraction: true
    cache_embeddings: true
```

**Performance Optimization**
```python
class OptimizedSpeakerProcessor:
    def __init__(self, config):
        self.config = config
        self.embedding_cache = {}
        self.model_cache = {}
    
    def batch_process_speakers(self, audio_files):
        """Process multiple files efficiently."""
        # Batch embedding extraction
        all_embeddings = self.batch_extract_embeddings(audio_files)
        
        # Parallel speaker identification
        with concurrent.futures.ThreadPoolExecutor() as executor:
            identification_futures = {
                executor.submit(self.identify_speakers, embeddings): file_id
                for file_id, embeddings in all_embeddings.items()
            }
            
            results = {}
            for future in concurrent.futures.as_completed(identification_futures):
                file_id = identification_futures[future]
                results[file_id] = future.result()
        
        return results
    
    def optimize_for_real_time(self):
        """Optimize system for real-time processing."""
        # Pre-load models
        self.preload_models()
        
        # Configure for low latency
        self.config.update({
            'embedding_batch_size': 1,
            'streaming_mode': True,
            'buffer_size': 1.0,  # 1 second buffer
            'overlap_processing': True
        })
    
    def preload_models(self):
        """Pre-load all models to reduce inference latency."""
        print("Pre-loading speaker models...")
        
        # Load diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            self.config['diarization_pipeline']
        )
        
        # Load embedding model
        self.embedding_model = PretrainedSpeakerEmbedding(
            self.config['embedding_model']
        )
        
        # Warm up models with dummy data
        dummy_audio = np.random.randn(16000)  # 1 second of dummy audio
        _ = self.embedding_model(dummy_audio)
        
        print("Models loaded and warmed up.")
```

This comprehensive speaker identification system provides robust speaker diarization, identification, and tracking capabilities while maintaining high accuracy and performance for real-world applications.