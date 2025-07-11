// Scripflow Upload Handler
class UploadHandler {
    constructor() {
        this.supportedFormats = [
            'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/aac', 'audio/ogg',
            'video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo'
        ];
        this.maxFileSize = 500 * 1024 * 1024; // 500MB
    }
    
    validateFile(file) {
        const errors = [];
        
        // Check file size
        if (file.size > this.maxFileSize) {
            errors.push(`File size (${this.formatFileSize(file.size)}) exceeds maximum allowed size (500MB)`);
        }
        
        // Check file format
        if (!this.supportedFormats.includes(file.type)) {
            errors.push(`File format "${file.type}" is not supported`);
        }
        
        // Check file name
        if (!file.name || file.name.trim() === '') {
            errors.push('File name is required');
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors
        };
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    async uploadFile(file, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            const formData = new FormData();
            formData.append('file', file);
            
            // Progress tracking
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    if (onProgress) {
                        onProgress(percentComplete);
                    }
                }
            });
            
            // Response handling
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            });
            
            // Error handling
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });
            
            xhr.addEventListener('abort', () => {
                reject(new Error('Upload was aborted'));
            });
            
            // Start upload
            xhr.open('POST', 'http://localhost:8000/api/upload');
            xhr.send(formData);
        });
    }
    
    createUploadProgressBar() {
        const progressContainer = document.createElement('div');
        progressContainer.className = 'fixed bottom-4 right-4 bg-white rounded-lg shadow-lg p-4 w-80 z-50';
        progressContainer.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium text-gray-800">Uploading...</span>
                <button class="text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="progress-bar h-2 rounded-full" style="width: 0%"></div>
            </div>
            <div class="text-xs text-gray-500 mt-1">
                <span class="progress-text">0%</span> • <span class="file-name"></span>
            </div>
        `;
        
        document.body.appendChild(progressContainer);
        return progressContainer;
    }
    
    updateProgressBar(container, percent, fileName) {
        const progressBar = container.querySelector('.progress-bar');
        const progressText = container.querySelector('.progress-text');
        const fileNameElement = container.querySelector('.file-name');
        
        progressBar.style.width = `${percent}%`;
        progressText.textContent = `${Math.round(percent)}%`;
        if (fileName) {
            fileNameElement.textContent = fileName;
        }
    }
    
    removeProgressBar(container) {
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    }
    
    // File type detection by extension
    getFileType(file) {
        const extension = file.name.split('.').pop().toLowerCase();
        const audioExtensions = ['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'];
        const videoExtensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'];
        
        if (audioExtensions.includes(extension)) {
            return 'audio';
        } else if (videoExtensions.includes(extension)) {
            return 'video';
        }
        return 'unknown';
    }
    
    // Get file duration (for supported formats)
    async getFileDuration(file) {
        return new Promise((resolve) => {
            if (file.type.startsWith('audio/')) {
                const audio = new Audio();
                audio.preload = 'metadata';
                
                audio.onloadedmetadata = () => {
                    resolve(audio.duration);
                };
                
                audio.onerror = () => {
                    resolve(null);
                };
                
                audio.src = URL.createObjectURL(file);
            } else {
                resolve(null);
            }
        });
    }
    
    // Create file preview
    createFilePreview(file) {
        const preview = document.createElement('div');
        preview.className = 'bg-white rounded-lg p-4 border border-gray-200 mb-4';
        
        const fileType = this.getFileType(file);
        const icon = fileType === 'audio' ? 'fa-music' : 'fa-video';
        
        preview.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                    <i class="fas ${icon} text-blue-600"></i>
                </div>
                <div class="flex-1">
                    <div class="font-medium text-gray-800">${file.name}</div>
                    <div class="text-sm text-gray-500">
                        ${this.formatFileSize(file.size)} • ${fileType.toUpperCase()}
                    </div>
                </div>
                <button class="text-red-500 hover:text-red-700" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        return preview;
    }
}

// Upload Management
class UploadManager {
    constructor() {
        this.dragDropArea = null;
        this.fileInput = null;
        this.uploadButton = null;
        this.loadingOverlay = null;
        
        this.initializeElements();
        this.initializeEventListeners();
    }

    initializeElements() {
        this.dragDropArea = document.getElementById('drag-drop-area');
        this.fileInput = document.getElementById('file-input');
        this.uploadButton = document.getElementById('open-files-btn');
        this.loadingOverlay = document.getElementById('loading-overlay');
        
        console.log('UploadManager elements initialized:');
        console.log('- dragDropArea:', this.dragDropArea);
        console.log('- fileInput:', this.fileInput);
        console.log('- uploadButton:', this.uploadButton);
        console.log('- loadingOverlay:', this.loadingOverlay);
    }

    initializeEventListeners() {
        // File input change - prevent duplicate handling
        this.fileInput.addEventListener('change', (e) => {
            console.log('File input changed, files selected:', e.target.files.length);
            // Clear the input to prevent duplicate processing
            const files = Array.from(e.target.files);
            e.target.value = '';
            this.handleFileSelection(files);
        });

        // Upload button click
        this.uploadButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            console.log('Upload button clicked, triggering file input');
            this.fileInput.click();
        });

        // Drag and drop events
        this.dragDropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dragDropArea.classList.add('drag-active');
        });

        this.dragDropArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.dragDropArea.classList.remove('drag-active');
        });

        this.dragDropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dragDropArea.classList.remove('drag-active');
            this.handleFileSelection(e.dataTransfer.files);
        });

        // Other feature buttons
        document.getElementById('new-recording-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('New Recording');
        });

        document.getElementById('record-meeting-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Record Meeting');
        });

        document.getElementById('batch-transcription-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Batch Transcription');
        });

        document.getElementById('record-app-audio-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Record App Audio');
        });

        document.getElementById('dictation-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Dictation');
        });

        document.getElementById('transcribe-podcast-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Transcribe Podcast');
        });

        document.getElementById('global-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Global');
        });

        document.getElementById('cloud-transcription-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Cloud Transcription');
        });

        document.getElementById('manage-models-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Manage Models');
        });

        document.getElementById('calendar-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Calendar');
        });

        document.getElementById('support-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Support');
        });

        document.getElementById('download-ios-btn').addEventListener('click', () => {
            this.showFeatureNotImplemented('Download iOS App');
        });

        // URL input
        document.getElementById('url-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleUrlInput();
            }
        });
    }

    handleFileSelection(files) {
        if (files.length === 0) return;

        const file = files[0]; // For now, handle only the first file
        
        // Validate file type
        if (!this.isValidAudioFile(file)) {
            this.showError('Please select a valid audio file (MP3, WAV, M4A, MP4, etc.)');
            return;
        }

        // Validate file size (100MB limit)
        if (file.size > 100 * 1024 * 1024) {
            this.showError('File size must be less than 100MB');
            return;
        }

        // Get configuration from UI
        const config = this.getTranscriptionConfig();
        
        // Start transcription process
        this.startTranscriptionProcess(file, config);
    }

    handleUrlInput() {
        const url = document.getElementById('url-input').value.trim();
        if (url) {
            this.showFeatureNotImplemented('URL Processing');
        }
    }

    isValidAudioFile(file) {
        const validTypes = [
            'audio/mpeg',
            'audio/wav',
            'audio/mp4',
            'audio/aac',
            'audio/ogg',
            'audio/webm',
            'video/mp4',
            'video/webm'
        ];
        
        return validTypes.includes(file.type) || 
               file.name.match(/\.(mp3|wav|m4a|mp4|aac|ogg|webm)$/i);
    }

    getTranscriptionConfig() {
        const language = document.getElementById('language-select').value;
        const engine = document.getElementById('engine-select').value;
        const model = document.getElementById('model-select').value;
        
        return {
            language: language === 'auto' ? 'auto' : language,
            transcription_engine: engine,
            whisper_model: model,  // Use selected model instead of hardcoded 'base'
            vad_method: 'simple',
            device: 'cpu',  // Changed from 'cuda' to 'cpu' for better compatibility
            enable_speaker_diarization: false,  // Disabled for faster processing
            show_romanized_text: false,
            single_speaker: true,  // Changed to true for faster processing
            preserve_speakers: false  // Changed to false for faster processing
        };
    }

    async startTranscriptionProcess(file, config) {
        try {
            // Show loading overlay
            this.showLoading('Uploading file...');
            
            // Start transcription through progress manager
            window.progressManager.startTranscription(file, config);
            
            // Hide loading overlay immediately - progress screen will handle the rest
            this.hideLoading();
            
        } catch (error) {
            this.hideLoading();
            this.showError('Failed to start transcription: ' + error.message);
        }
    }

    showLoading(message = 'Processing...') {
        console.log('🔵 showLoading called with message:', message);
        const loadingText = document.getElementById('loading-text');
        if (loadingText) {
            loadingText.textContent = message;
            console.log('✅ Loading text updated');
        } else {
            console.error('❌ Loading text element not found');
        }
        
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.remove('hidden');
            console.log('✅ Loading overlay shown');
        } else {
            console.error('❌ Loading overlay element not found');
        }
    }

    hideLoading() {
        console.log('🔴 hideLoading called');
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('hidden');
            console.log('✅ Loading overlay hidden');
        } else {
            console.error('❌ Loading overlay element not found in hideLoading');
        }
    }

    showError(message) {
        // Create a simple error notification
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    showFeatureNotImplemented(featureName) {
        // Create a notification for features not yet implemented
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-blue-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        notification.innerHTML = `
            <div class="flex items-center gap-2">
                <i class="fas fa-info-circle"></i>
                <span>${featureName} will be available in Phase 3</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 4 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 4000);
    }

    // Utility function to format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Simple, clean initialization - REMOVED DUPLICATE EVENT LISTENERS
// The UploadManager class handles all event listeners properly

// Initialize other components
document.addEventListener('DOMContentLoaded', function() {
    // Initialize upload handler
    window.uploadHandler = new UploadHandler();
    
    // Initialize upload manager (only once)
    if (!window.uploadManager) {
        window.uploadManager = new UploadManager();
        window.uploadManager.initializeElements();
        window.uploadManager.initializeEventListeners();
    }
}); 