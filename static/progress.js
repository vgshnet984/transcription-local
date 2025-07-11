// Progress Screen Management
class ProgressManager {
    constructor() {
        this.currentJobId = null;
        this.progressInterval = null;
        this.startTime = null;
        this.isCancelled = false;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Navigation buttons
        document.getElementById('back-to-upload').addEventListener('click', () => {
            this.cancelTranscription();
            this.showScreen(1);
        });

        document.getElementById('cancel-transcription').addEventListener('click', () => {
            this.cancelTranscription();
        });

        // Back to progress from completed screen
        document.getElementById('back-to-progress').addEventListener('click', () => {
            this.showScreen(2);
        });

        // Add manual check for completed jobs
        const checkCompletedBtn = document.getElementById('check-completed');
        if (checkCompletedBtn) {
            checkCompletedBtn.addEventListener('click', () => {
                this.checkForCompletedJobs();
            });
        }
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

    startTranscription(fileData, config) {
        // Prevent duplicate uploads
        if (this.currentJobId !== null) {
            console.log('Transcription already in progress, ignoring duplicate request');
            return;
        }
        
        this.currentJobId = null;
        this.startTime = Date.now();
        this.isCancelled = false;
        
        // Update progress screen with file info
        this.updateFileInfo(fileData);
        this.updateEngineInfo(config);
        
        // Show progress screen
        this.showScreen(2);
        
        // Hide loading overlay if it exists
        console.log('Attempting to hide loading overlay...');
        console.log('window.uploadManager:', window.uploadManager);
        console.log('window.uploadManager?.loadingOverlay:', window.uploadManager?.loadingOverlay);
        
        if (window.uploadManager && window.uploadManager.loadingOverlay) {
            console.log('Hiding loading overlay via uploadManager');
            window.uploadManager.hideLoading();
        } else {
            console.log('UploadManager not available, trying direct approach');
            // Fallback: try to hide loading overlay directly
            const loadingOverlay = document.getElementById('loading-overlay');
            if (loadingOverlay) {
                console.log('Hiding loading overlay directly');
                loadingOverlay.classList.add('hidden');
            } else {
                console.log('Loading overlay element not found');
            }
        }
        
        // Start progress tracking
        this.startProgressTracking();
        
        // Upload file and start transcription
        this.uploadAndTranscribe(fileData, config);
    }

    updateFileInfo(fileData) {
        const filename = document.getElementById('progress-filename');
        const fileinfo = document.getElementById('progress-fileinfo');
        
        filename.textContent = fileData.name;
        fileinfo.textContent = `${this.formatFileSize(fileData.size)} • ${fileData.type}`;
    }

    updateEngineInfo(config) {
        const engine = document.getElementById('progress-engine');
        const engineNames = {
            'faster-whisper': 'Fast-Whisper',
            'whisperx': 'WhisperX',
            'parakeet': 'NVIDIA Parakeet',
            'whisper': 'Whisper'
        };
        engine.textContent = engineNames[config.transcription_engine] || config.transcription_engine;
    }

    async uploadAndTranscribe(fileData, config) {
        try {
            // Add status log entry
            this.addStatusLog('Uploading file...', 'info');
            console.log('Starting uploadAndTranscribe with file:', fileData.name, 'size:', fileData.size);
            
            // Upload file
            console.log('Calling uploadFile...');
            const uploadResponse = await this.uploadFile(fileData);
            console.log('Upload response received:', uploadResponse);
            
            if (uploadResponse.id) {  // Check for id instead of success
                this.addStatusLog('File uploaded successfully', 'success');
                this.addStatusLog('Starting transcription...', 'info');
                
                // Start transcription
                console.log('Calling startTranscriptionJob with file_id:', uploadResponse.id);
                const transcribeResponse = await this.startTranscriptionJob(uploadResponse.id, config);
                console.log('Transcribe response received:', transcribeResponse);
                
                if (transcribeResponse.transcription_id) {  // Check for transcription_id
                    this.currentJobId = transcribeResponse.transcription_id;  // Use transcription_id as job_id
                    console.log('🎯 Set currentJobId to transcription_id:', this.currentJobId);
                    this.addStatusLog('Transcription job started', 'success');
                    this.addStatusLog('Processing audio...', 'info');
                } else if (transcribeResponse.error) {
                    this.addStatusLog('Failed to start transcription: ' + transcribeResponse.error, 'error');
                } else {
                    // Assume it's a job ID if no transcription_id (new async format)
                    this.currentJobId = transcribeResponse.transcription_id || transcribeResponse.job_id || transcribeResponse.id;
                    console.log('🎯 Set currentJobId to job ID:', this.currentJobId, 'from response:', transcribeResponse);
                    this.addStatusLog('Transcription job started in background', 'success');
                    this.addStatusLog('Processing audio...', 'info');
                }
            } else {
                this.addStatusLog('Failed to upload file: ' + (uploadResponse.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Error in uploadAndTranscribe:', error);
            this.addStatusLog('Error: ' + error.message, 'error');
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        console.log('Starting upload...');
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', response.status);
        const result = await response.json();
        console.log('Upload response:', result);
        return result;
    }

    async startTranscriptionJob(fileId, config) {
        console.log('🚀 startTranscriptionJob called with config:', config);
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: fileId,
                config: config  // Send the entire config object
            })
        });

        const result = await response.json();
        console.log('📡 Transcription job response:', result);
        return result;
    }

    startProgressTracking() {
        console.log('🔄 Starting progress tracking with job ID:', this.currentJobId);
        
        // Update elapsed time every second
        setInterval(() => {
            if (this.startTime && !this.isCancelled) {
                this.updateElapsedTime();
            }
        }, 1000);

        // Check job status every 2 seconds
        this.progressInterval = setInterval(() => {
            if (this.currentJobId && !this.isCancelled) {
                console.log('⏰ Progress check triggered for job ID:', this.currentJobId);
                this.checkJobStatus();
            } else {
                console.log('⚠️ Progress check skipped - currentJobId:', this.currentJobId, 'isCancelled:', this.isCancelled);
            }
        }, 2000);
    }

    updateElapsedTime() {
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        document.getElementById('elapsed-time').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    async checkJobStatus() {
        try {
            console.log('🔍 Checking job status for job ID:', this.currentJobId);
            const response = await fetch(`/api/jobs/${this.currentJobId}`);
            const jobData = await response.json();
            console.log('📡 Job status check response:', jobData);

            if (jobData.success && jobData.job) {
                const job = jobData.job;
                console.log('📊 Job status:', job.status, 'Progress:', job.progress, 'Job ID:', job.id);
                this.updateProgress(job.progress, job.status);
                
                if (job.status === 'completed') {
                    console.log('✅ Job completed, calling handleTranscriptionComplete');
                    this.handleTranscriptionComplete(job);
                } else if (job.status === 'failed') {
                    console.log('❌ Job failed:', job.error_message);
                    this.handleTranscriptionFailed(job.error_message);
                }
            } else {
                console.error('❌ Invalid job data response:', jobData);
            }
        } catch (error) {
            console.error('❌ Error checking job status:', error);
        }
    }

    updateProgress(progress, status) {
        // Update progress ring
        const circle = document.getElementById('progress-circle');
        const percentage = document.getElementById('progress-percentage');
        const statusText = document.getElementById('progress-status');
        
        const circumference = 2 * Math.PI * 52; // r=52
        const offset = circumference - (progress / 100) * circumference;
        
        circle.style.strokeDashoffset = offset;
        percentage.textContent = `${Math.round(progress)}%`;
        
        // Update status text
        const statusMessages = {
            'processing': 'Transcribing audio...',
            'completed': 'Transcription complete!',
            'failed': 'Transcription failed',
            'cancelled': 'Transcription cancelled'
        };
        statusText.textContent = statusMessages[status] || status;

        // Update speakers info
        if (progress > 10) {
            document.getElementById('progress-speakers').textContent = '2 speakers detected';
        }
    }

    handleTranscriptionComplete(job) {
        console.log('🎉 handleTranscriptionComplete called with job:', job);
        this.addStatusLog('Transcription completed successfully!', 'success');
        this.updateProgress(100, 'completed');
        
        // Stop progress tracking
        if (this.progressInterval) {
            console.log('🛑 Stopping progress tracking');
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Navigate to completed screen after a short delay
        setTimeout(() => {
            console.log('🚀 Calling showCompletedScreen with job:', job);
            this.showCompletedScreen(job);
        }, 1500);
    }

    handleTranscriptionFailed(errorMessage) {
        this.addStatusLog('Transcription failed: ' + errorMessage, 'error');
        this.updateProgress(0, 'failed');
        
        // Stop progress tracking
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    async showCompletedScreen(job) {
        console.log('showCompletedScreen called with job:', job);
        // Get transcription details
        try {
            const response = await fetch(`/api/transcriptions/${job.transcription_id}`);
            const transcriptionData = await response.json();
            console.log('Transcription data received:', transcriptionData);
            
            if (transcriptionData && transcriptionData.id) {
                console.log('✅ Transcription data is valid, initializing transcript manager...');
                console.log('📄 Transcription data structure:', {
                    id: transcriptionData.id,
                    text: transcriptionData.text?.substring(0, 100) + '...',
                    segments: transcriptionData.segments?.length || 0,
                    confidence: transcriptionData.confidence
                });
                console.log('🎵 Job data structure:', {
                    id: job.id,
                    audio_file: job.audio_file,
                    transcription_id: job.transcription_id
                });
                
                // Initialize transcript screen with data
                window.transcriptManager.initializeTranscription(transcriptionData, job);
                console.log('✅ Transcript manager initialized, showing screen 3...');
                this.showScreen(3);
                console.log('✅ Screen 3 should now be visible');
            } else {
                console.error('❌ Transcription data error:', transcriptionData);
                console.error('❌ Job data:', job);
                this.addStatusLog('Error: Invalid transcription data received', 'error');
            }
        } catch (error) {
            console.error('Error loading transcription:', error);
            this.addStatusLog('Error loading transcription results: ' + error.message, 'error');
        }
    }

    cancelTranscription() {
        this.isCancelled = true;
        
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        this.addStatusLog('Transcription cancelled', 'warning');
        this.updateProgress(0, 'cancelled');
        
        // TODO: Implement actual job cancellation API call
        // if (this.currentJobId) {
        //     fetch(`/api/jobs/${this.currentJobId}/cancel`, { method: 'POST' });
        // }
    }

    addStatusLog(message, type = 'info') {
        const statusLog = document.getElementById('status-log');
        const logEntry = document.createElement('div');
        logEntry.className = 'flex items-center gap-2 text-sm';
        
        const dot = document.createElement('div');
        dot.className = 'w-2 h-2 rounded-full';
        
        switch (type) {
            case 'success':
                dot.className += ' bg-green-500';
                break;
            case 'error':
                dot.className += ' bg-red-500';
                break;
            case 'warning':
                dot.className += ' bg-yellow-500';
                break;
            default:
                dot.className += ' bg-blue-500';
        }
        
        const text = document.createElement('span');
        text.className = 'text-gray-600';
        text.textContent = message;
        
        logEntry.appendChild(dot);
        logEntry.appendChild(text);
        
        statusLog.appendChild(logEntry);
        
        // Auto-scroll to bottom
        statusLog.scrollTop = statusLog.scrollHeight;
        
        // Limit log entries
        while (statusLog.children.length > 10) {
            statusLog.removeChild(statusLog.firstChild);
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async checkForCompletedJobs() {
        try {
            console.log('🔍 Checking for completed jobs...');
            const response = await fetch('/api/jobs');
            const jobs = await response.json();
            
            // Find the latest completed job
            const completedJob = jobs.find(job => job.status === 'completed');
            
            if (completedJob) {
                console.log('✅ Found completed job:', completedJob.id);
                this.addStatusLog(`Found completed job ${completedJob.id}, loading results...`, 'success');
                this.handleTranscriptionComplete(completedJob);
            } else {
                console.log('❌ No completed jobs found');
                this.addStatusLog('No completed jobs found', 'warning');
            }
        } catch (error) {
            console.error('❌ Error checking for completed jobs:', error);
            this.addStatusLog('Error checking for completed jobs: ' + error.message, 'error');
        }
    }
}

// Initialize progress manager
window.progressManager = new ProgressManager(); 