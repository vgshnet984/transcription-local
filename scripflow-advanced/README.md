# Scripflow - Advanced Transcription Interface

## Overview

Scripflow is an advanced, modern transcription interface built on top of the existing transcription platform. It provides a Windows 11-style interface with enhanced features while maintaining compatibility with the existing backend.

## Features

### ✅ **Implemented (Phase 1)**
- **Modern Windows 11-style UI** with gradient backgrounds and smooth animations
- **File Upload Interface** with drag & drop support
- **Multiple Upload Methods**:
  - Open Files button
  - Drag & drop zone
  - URL input (placeholder)
- **Engine Selection** with dropdown menus
- **Language Selection** with auto-detect option
- **History Sidebar** with searchable transcription history
- **Real-time Progress Tracking** with loading overlays
- **Toast Notifications** for user feedback
- **Responsive Design** for different screen sizes

### 🚧 **Coming Soon (Phase 2)**
- **Transcription Progress Screen** with real-time updates
- **Advanced Audio Player** with waveform visualization
- **Speaker Diarization Visualization** with speaker timeline
- **Transcript Editor** with inline editing capabilities
- **Export Options** (SRT, VTT, TXT, JSON)
- **Batch Processing** interface
- **Recording Features** (microphone, meeting, app audio)
- **Settings Panel** for advanced configuration

### 🔮 **Future Features (Phase 3)**
- **Cloud Transcription** integration
- **Podcast Transcription** with episode management
- **Dictation Mode** for real-time speech-to-text
- **Calendar Integration** for scheduled transcriptions
- **Model Management** interface
- **iOS App** download and sync
- **Support System** integration

## Access

- **Current Interface**: `http://localhost:8000/` (original interface)
- **Scripflow Interface**: `http://localhost:8000/scripflow` (new advanced interface)

## Architecture

```
frontend/scripflow/
├── index.html          # Main interface
├── css/
│   └── main.css        # Custom styles and animations
├── js/
│   ├── app.js          # Main application logic
│   └── upload.js       # File upload handling
├── assets/
│   └── icons/          # Custom icons (future)
└── README.md           # This file
```

## Development Status

### Phase 1: Foundation ✅
- [x] Basic UI structure
- [x] File upload functionality
- [x] API integration
- [x] Progress tracking
- [x] Error handling

### Phase 2: Core Features 🚧
- [ ] Transcription progress screen
- [ ] Audio player integration
- [ ] Speaker visualization
- [ ] Transcript editing
- [ ] Export functionality

### Phase 3: Advanced Features 📋
- [ ] Recording capabilities
- [ ] Batch processing
- [ ] Cloud integration
- [ ] Advanced settings

## Technical Details

### Backend Integration
- Uses existing API endpoints at `http://localhost:8000/api/`
- Maintains full compatibility with current backend
- No changes required to existing functionality

### Frontend Technologies
- **HTML5** with semantic markup
- **Tailwind CSS** for styling
- **Vanilla JavaScript** (ES6+) for functionality
- **FontAwesome** for icons
- **Custom CSS** for Windows 11 styling

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Usage

1. **Start the server** with the existing backend
2. **Navigate to** `http://localhost:8000/scripflow`
3. **Upload files** using drag & drop or the "Open Files" button
4. **Configure settings** using the dropdown menus
5. **Monitor progress** through the interface
6. **Access history** through the sidebar

## Development Notes

- The interface is designed to be **non-intrusive** to existing functionality
- All new features are **additive** and don't modify existing code
- The backend remains **unchanged** and fully functional
- Both interfaces can run **simultaneously** without conflicts

## Contributing

When adding new features:
1. Keep the existing interface intact
2. Add new functionality as separate components
3. Maintain backward compatibility
4. Follow the established design patterns
5. Test both interfaces after changes 