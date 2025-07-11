// Transcript Screen Management
class TranscriptManager {
    constructor() {
        this.currentTranscription = null;
        this.currentJob = null;
        this.audioPlayer = null;
        this.isPlaying = false;
        this.currentTime = 0;
        this.duration = 0;
        this.playbackSpeed = 1;
        this.showSpeakers = true;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Audio player controls
        document.getElementById('play-pause-btn').addEventListener('click', () => {
            this.togglePlayback();
        });

        document.getElementById('speed-btn').addEventListener('click', () => {
            this.cyclePlaybackSpeed();
        });

        document.getElementById('rewind-btn').addEventListener('click', () => {
            this.seek(-10);
        });

        document.getElementById('forward-btn').addEventListener('click', () => {
            this.seek(10);
        });

        // Search functionality
        document.getElementById('search-btn').addEventListener('click', () => {
            this.toggleSearch();
        });

        document.getElementById('transcript-search').addEventListener('input', (e) => {
            this.searchTranscript(e.target.value);
        });

        // Speaker toggle
        document.getElementById('speaker-toggle').addEventListener('click', () => {
            this.toggleSpeakers();
        });

        // Export functionality
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportTranscription();
        });

        // Edit functionality
        document.getElementById('edit-btn').addEventListener('click', () => {
            this.enableEditing();
        });

        // New transcription functionality
        document.getElementById('new-transcription-btn').addEventListener('click', () => {
            this.startNewTranscription();
        });
    }

    initializeTranscription(transcription, job) {
        console.log('🎯 initializeTranscription called with:', { transcription, job });
        this.currentTranscription = transcription;
        this.currentJob = job;
        
        console.log('📝 Step 1: Updating file info...');
        // Update file info
        this.updateFileInfo();
        
        console.log('📊 Step 2: Updating statistics...');
        // Update statistics
        this.updateStatistics();
        
        console.log('🎵 Step 3: Loading audio file...');
        // Load audio file
        this.loadAudioFile();
        
        console.log('📄 Step 4: Displaying transcript...');
        // Display transcript
        this.displayTranscript();
        console.log('✅ initializeTranscription completed successfully');
    }

    updateFileInfo() {
        console.log('📝 updateFileInfo called');
        const filename = document.getElementById('completed-filename');
        const fileinfo = document.getElementById('completed-fileinfo');
        
        console.log('🔍 Looking for elements:', {
            filenameElement: filename,
            fileinfoElement: fileinfo
        });
        
        // Get file info from job or transcription
        const audioFile = this.currentJob.audio_file;
        console.log('🎵 Audio file data:', audioFile);
        
        if (audioFile) {
            filename.textContent = audioFile.filename;
            fileinfo.textContent = `${this.formatFileSize(audioFile.size)} • ${this.formatDuration(audioFile.duration)}`;
            console.log('✅ File info updated:', {
                filename: audioFile.filename,
                size: this.formatFileSize(audioFile.size),
                duration: this.formatDuration(audioFile.duration)
            });
        } else {
            console.error('❌ No audio file data found');
        }
    }

    updateStatistics() {
        const duration = document.getElementById('transcription-duration');
        const words = document.getElementById('transcription-words');
        const confidence = document.getElementById('transcription-confidence');
        const speakers = document.getElementById('transcription-speakers');
        
        // Calculate statistics
        const wordCount = this.currentTranscription.text.split(/\s+/).length;
        const confidencePercent = Math.round(this.currentTranscription.confidence * 100);
        
        // Count unique speakers
        const speakerSet = new Set();
        if (this.currentTranscription.segments) {
            this.currentTranscription.segments.forEach(segment => {
                if (segment.speaker_id) {
                    speakerSet.add(segment.speaker_id);
                }
            });
        }
        const speakerCount = speakerSet.size || 1;
        
        duration.textContent = this.formatDuration(this.currentJob.audio_file?.duration || 0);
        words.textContent = wordCount.toLocaleString();
        confidence.textContent = `${confidencePercent}%`;
        speakers.textContent = speakerCount;
    }

    async loadAudioFile() {
        console.log('🎵 loadAudioFile called');
        const audioElement = document.getElementById('audio-player');
        console.log('🔍 Audio element:', audioElement);
        
        // Get audio file URL from job
        if (this.currentJob.audio_file) {
            console.log('🎵 Audio file data:', this.currentJob.audio_file);
            
            // Use the actual file path from the backend, or fallback to filename
            const filePath = this.currentJob.audio_file.path || this.currentJob.audio_file.filename;
            console.log('📁 File path:', filePath);
            
            // Extract just the filename from the path (remove 'uploads/' prefix if present)
            const fileName = filePath.replace(/^uploads[\/\\]/, '');
            const audioUrl = `/uploads/${fileName}`;
            console.log('🔗 Audio URL:', audioUrl);
            
            audioElement.src = audioUrl;
            console.log('✅ Audio source set to:', audioUrl);
            
            audioElement.addEventListener('loadedmetadata', () => {
                this.duration = audioElement.duration;
                document.getElementById('total-time').textContent = this.formatTime(this.duration);
                console.log('✅ Audio metadata loaded, duration:', this.duration);
            });
            
            audioElement.addEventListener('timeupdate', () => {
                this.currentTime = audioElement.currentTime;
                this.updateAudioProgress();
            });
            
            audioElement.addEventListener('ended', () => {
                this.isPlaying = false;
                this.updatePlayButton();
            });
            
            audioElement.addEventListener('error', (e) => {
                console.error('❌ Audio loading error:', e);
            });
        } else {
            console.error('❌ No audio file data found in job');
        }
    }

    displayTranscript() {
        console.log('📄 displayTranscript called');
        const content = document.getElementById('transcript-content');
        console.log('🔍 Transcript content element:', content);
        
        console.log('📝 Current transcription data:', {
            text: this.currentTranscription.text?.substring(0, 100) + '...',
            segments: this.currentTranscription.segments?.length || 0,
            confidence: this.currentTranscription.confidence
        });
        
        if (!this.currentTranscription.segments || this.currentTranscription.segments.length === 0) {
            console.log('📄 No segments found, displaying full text');
            // Display full text if no segments
            content.innerHTML = `
                <div class="bg-gray-50 rounded-lg p-4">
                    <div class="text-gray-800 leading-relaxed">
                        ${this.currentTranscription.text}
                    </div>
                </div>
            `;
            console.log('✅ Full text displayed');
        } else {
            console.log('📄 Displaying segmented transcript');
            // Display segmented transcript
            content.innerHTML = this.currentTranscription.segments.map(segment => {
                const speakerClass = this.getSpeakerClass(segment.speaker_id);
                const speakerPill = this.showSpeakers ? 
                    `<span class="speaker-pill ${speakerClass}">${segment.speaker_id}</span>` : '';
                
                return `
                    <div class="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors cursor-pointer" 
                         data-start="${segment.start_time}" data-end="${segment.end_time}">
                        <div class="flex items-start gap-3">
                            <div class="flex-shrink-0">
                                <div class="text-xs text-gray-500 font-mono">
                                    ${this.formatTime(segment.start_time)}
                                </div>
                            </div>
                            <div class="flex-1">
                                <div class="flex items-center gap-2 mb-2">
                                    ${speakerPill}
                                    <span class="text-xs text-gray-500">
                                        Confidence: ${Math.round(segment.confidence * 100)}%
                                    </span>
                                </div>
                                <div class="text-gray-800 leading-relaxed">
                                    ${segment.text}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Add click handlers for segments
            content.querySelectorAll('[data-start]').forEach(segment => {
                segment.addEventListener('click', () => {
                    const startTime = parseFloat(segment.dataset.start);
                    this.seekToTime(startTime);
                });
            });
            console.log('✅ Segmented transcript displayed');
        }
    }

    togglePlayback() {
        const audioElement = document.getElementById('audio-player');
        
        if (this.isPlaying) {
            audioElement.pause();
        } else {
            audioElement.play();
        }
        
        this.isPlaying = !this.isPlaying;
        this.updatePlayButton();
    }

    updatePlayButton() {
        const button = document.getElementById('play-pause-btn');
        const icon = button.querySelector('i');
        
        if (this.isPlaying) {
            icon.className = 'fas fa-pause';
        } else {
            icon.className = 'fas fa-play';
        }
    }

    updateAudioProgress() {
        const progressBar = document.getElementById('audio-progress');
        const currentTimeDisplay = document.getElementById('current-time');
        
        const progress = (this.currentTime / this.duration) * 100;
        progressBar.style.width = `${progress}%`;
        currentTimeDisplay.textContent = this.formatTime(this.currentTime);
    }

    seek(seconds) {
        const audioElement = document.getElementById('audio-player');
        const newTime = Math.max(0, Math.min(this.duration, this.currentTime + seconds));
        audioElement.currentTime = newTime;
    }

    seekToTime(time) {
        const audioElement = document.getElementById('audio-player');
        audioElement.currentTime = time;
    }

    cyclePlaybackSpeed() {
        const speeds = [0.5, 0.75, 1, 1.25, 1.5, 2];
        const currentIndex = speeds.indexOf(this.playbackSpeed);
        const nextIndex = (currentIndex + 1) % speeds.length;
        this.playbackSpeed = speeds[nextIndex];
        
        const audioElement = document.getElementById('audio-player');
        audioElement.playbackRate = this.playbackSpeed;
        
        document.getElementById('speed-btn').textContent = `${this.playbackSpeed}x`;
    }

    toggleSearch() {
        const searchBar = document.getElementById('search-bar');
        searchBar.classList.toggle('hidden');
        
        if (!searchBar.classList.contains('hidden')) {
            document.getElementById('transcript-search').focus();
        }
    }

    searchTranscript(query) {
        const segments = document.querySelectorAll('#transcript-content > div');
        
        segments.forEach(segment => {
            const text = segment.textContent.toLowerCase();
            const matches = text.includes(query.toLowerCase());
            
            if (query === '') {
                segment.style.opacity = '1';
                segment.style.backgroundColor = '';
            } else if (matches) {
                segment.style.opacity = '1';
                segment.style.backgroundColor = '#fef3c7';
            } else {
                segment.style.opacity = '0.3';
                segment.style.backgroundColor = '';
            }
        });
    }

    toggleSpeakers() {
        this.showSpeakers = !this.showSpeakers;
        this.displayTranscript();
        
        const button = document.getElementById('speaker-toggle');
        const icon = button.querySelector('i');
        
        if (this.showSpeakers) {
            icon.className = 'fas fa-users';
        } else {
            icon.className = 'fas fa-user';
        }
    }

    exportTranscription() {
        if (!this.currentTranscription) return;
        
        const formats = [
            { name: 'Text (.txt)', extension: 'txt', mime: 'text/plain' },
            { name: 'SRT (.srt)', extension: 'srt', mime: 'text/plain' },
            { name: 'VTT (.vtt)', extension: 'vtt', mime: 'text/plain' },
            { name: 'JSON (.json)', extension: 'json', mime: 'application/json' }
        ];
        
        // For now, export as text
        const content = this.formatTranscriptForExport('txt');
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcript_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    formatTranscriptForExport(format) {
        if (format === 'txt') {
            return this.currentTranscription.text;
        } else if (format === 'srt') {
            return this.formatAsSRT();
        } else if (format === 'vtt') {
            return this.formatAsVTT();
        } else if (format === 'json') {
            return JSON.stringify(this.currentTranscription, null, 2);
        }
        return this.currentTranscription.text;
    }

    formatAsSRT() {
        if (!this.currentTranscription.segments) {
            return this.currentTranscription.text;
        }
        
        return this.currentTranscription.segments.map((segment, index) => {
            const startTime = this.formatTimeForSRT(segment.start_time);
            const endTime = this.formatTimeForSRT(segment.end_time);
            const speaker = segment.speaker_id ? `[${segment.speaker_id}] ` : '';
            
            return `${index + 1}\n${startTime} --> ${endTime}\n${speaker}${segment.text}\n`;
        }).join('\n');
    }

    formatAsVTT() {
        if (!this.currentTranscription.segments) {
            return this.currentTranscription.text;
        }
        
        let vtt = 'WEBVTT\n\n';
        
        this.currentTranscription.segments.forEach((segment, index) => {
            const startTime = this.formatTimeForVTT(segment.start_time);
            const endTime = this.formatTimeForVTT(segment.end_time);
            const speaker = segment.speaker_id ? `<${segment.speaker_id}> ` : '';
            
            vtt += `${index + 1}\n${startTime} --> ${endTime}\n${speaker}${segment.text}\n\n`;
        });
        
        return vtt;
    }

    formatTimeForSRT(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
    }

    formatTimeForVTT(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
    }

    enableEditing() {
        // TODO: Implement transcript editing functionality
        alert('Transcript editing will be implemented in Phase 3');
    }

    startNewTranscription() {
        console.log('🔄 Starting new transcription...');
        // Reset any current state
        this.currentTranscription = null;
        this.currentJob = null;
        
        // Stop any audio playback
        const audioElement = document.getElementById('audio-player');
        if (audioElement) {
            audioElement.pause();
            audioElement.currentTime = 0;
        }
        
        // Go back to the upload screen (Screen 1)
        this.showScreen(1);
        console.log('✅ Navigated to upload screen');
    }

    showScreen(screenNumber) {
        console.log('showScreen called with screenNumber:', screenNumber);
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        const targetScreen = document.getElementById(`screen-${screenNumber}`);
        if (targetScreen) {
            targetScreen.classList.add('active');
            console.log('Screen', screenNumber, 'activated');
        } else {
            console.error('Screen', screenNumber, 'not found');
        }
    }

    getSpeakerClass(speakerId) {
        const speakerNumber = speakerId.match(/\d+/);
        if (speakerNumber) {
            return `speaker-${speakerNumber[0]}`;
        }
        return 'speaker-1';
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize transcript manager
window.transcriptManager = new TranscriptManager(); 