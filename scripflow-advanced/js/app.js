// Scripflow Main Application
class ScripflowApp {
    constructor() {
        this.currentScreen = 'upload';
        this.currentTranscription = null;
        this.audioPlayer = null;
        this.speakers = [];
        this.history = [];
        this.apiBaseUrl = 'http://localhost:8000/api';
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadHistory();
        this.checkEngineStatus();
    }
    
    bindEvents() {
        // Remove all file upload and drag-drop event listeners from here.
        // Only keep non-upload UI logic.
        
        // URL input
        document.getElementById('url-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleUrlInput();
            }
        });
        
        // History search
        document.getElementById('history-search').addEventListener('input', (e) => {
            this.filterHistory(e.target.value);
        });
        
        // Settings button
        document.getElementById('settings-btn').addEventListener('click', () => {
            this.showSettings();
        });
        
        // Other feature buttons (placeholder for future features)
        this.bindFeatureButtons();
    }
    
    bindFeatureButtons() {
        const buttons = {
            'new-recording-btn': () => this.showToast('Recording feature coming soon!', 'warning'),
            'record-meeting-btn': () => this.showToast('Meeting recording feature coming soon!', 'warning'),
            'batch-transcription-btn': () => this.showToast('Batch transcription feature coming soon!', 'warning'),
            'record-app-audio-btn': () => this.showToast('App audio recording feature coming soon!', 'warning'),
            'dictation-btn': () => this.showToast('Dictation feature coming soon!', 'warning'),
            'transcribe-podcast-btn': () => this.showToast('Podcast transcription feature coming soon!', 'warning'),
            'global-btn': () => this.showToast('Global features coming soon!', 'warning'),
            'cloud-transcription-btn': () => this.showToast('Cloud transcription feature coming soon!', 'warning'),
            'manage-models-btn': () => this.showToast('Model management feature coming soon!', 'warning'),
            'calendar-btn': () => this.showToast('Calendar feature coming soon!', 'warning'),
            'support-btn': () => this.showToast('Support feature coming soon!', 'warning'),
            'download-ios-btn': () => this.showToast('iOS app coming soon!', 'warning')
        };
        
        Object.entries(buttons).forEach(([id, handler]) => {
            const button = document.getElementById(id);
            if (button) {
                button.addEventListener('click', handler);
            }
        });
    }
    
    // File handling is now done by upload.js and progress.js
    
    async handleUrlInput() {
        const url = document.getElementById('url-input').value.trim();
        if (!url) return;
        
        this.showToast('URL processing feature coming soon!', 'warning');
        // TODO: Implement URL processing
    }
    
    // Transcription handling is now done by progress.js
    
    // Transcription progress is now handled by progress.js
    
    async loadHistory() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/transcriptions`);
            if (response.ok) {
                const data = await response.json();
                this.history = data.transcriptions || [];
                this.renderHistory();
            }
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
    
    renderHistory() {
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';
        
        this.history.slice(0, 10).forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'flex items-center gap-2 p-2 bg-white rounded hover:bg-gray-50 cursor-pointer';
            historyItem.innerHTML = `
                <div class="status-indicator status-${item.status}"></div>
                <div class="flex-1 min-w-0">
                    <div class="text-xs font-medium text-gray-800 truncate">${item.filename || 'Untitled'}</div>
                    <div class="text-xs text-gray-500">${this.formatTime(item.created_at)}</div>
                </div>
            `;
            historyItem.addEventListener('click', () => this.openHistoryItem(item.id));
            historyList.appendChild(historyItem);
        });
    }
    
    filterHistory(searchTerm) {
        // TODO: Implement history filtering
        console.log('Filtering history:', searchTerm);
    }
    
    openHistoryItem(id) {
        // TODO: Implement opening history item
        this.showToast('History item opening feature coming soon!', 'warning');
    }
    
    async checkEngineStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/engine/info`);
            if (response.ok) {
                const data = await response.json();
                console.log('Engine status:', data);
                // TODO: Update UI based on engine status
            }
        } catch (error) {
            console.error('Failed to check engine status:', error);
        }
    }
    
    showSettings() {
        this.showToast('Settings panel coming soon!', 'warning');
    }
    
    showLoading(text = 'Processing...') {
        document.getElementById('loading-text').textContent = text;
        document.getElementById('loading-overlay').classList.remove('hidden');
    }
    
    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="flex items-center gap-2">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
    
    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
    
    formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.scripflowApp = new ScripflowApp();
}); 