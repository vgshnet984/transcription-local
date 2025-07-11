# GitHub Upload Guide for Transcription Project

## What Will Be Uploaded

### ✅ Included Files (Code & Configuration)
- All Python source code (`src/`, `scripts/`, `tests/`)
- Configuration files (`config.py`, `config.json`, `requirements*.txt`)
- Documentation (`README.md`, `*.md` files)
- Frontend files (`static/`, `templates/`, `frontend/`, `scripflow-advanced/`)
- Docker files (`Dockerfile.local`, `docker-compose.local.yml`)
- Batch scripts (`*.bat` files)
- Test scripts (all `test_*.py` files)

### ❌ Excluded Files (Large/Generated)
- `models/` directory (downloaded ML models - too large)
- `uploads/` directory (user uploaded audio files)
- `transcripts/` directory (generated transcripts)
- `logs/` directory (runtime logs)
- `*.db` files (SQLite databases)
- `*.wav`, `*.mp3` files (audio files)
- `__pycache__/` directories
- `.pytest_cache/` directory

## Step-by-Step Upload Process

### 1. Initialize Git Repository (Already Done)
```bash
git status  # Check current status
```

### 2. Add Files to Git
```bash
# Add all files except those in .gitignore
git add .

# Check what will be committed
git status
```

### 3. Create Initial Commit
```bash
git commit -m "Initial commit: Transcription platform with optimized WhisperX and Faster-Whisper integration"
```

### 4. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `transcription-local-optimized` (or your preferred name)
3. Description: "Local-first transcription platform with WhisperX, Faster-Whisper, and speaker diarization"
4. Make it Public or Private (your choice)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 5. Connect to GitHub Repository
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/transcription-local-optimized.git

# Set the main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

## Authentication Options

### Option 1: Personal Access Token (Recommended)
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` permissions
3. Copy the token
4. When prompted for password, use the token instead

### Option 2: GitHub CLI (Easiest)
```bash
# Install GitHub CLI if not installed
# Then authenticate
gh auth login
```

### Option 3: SSH Keys
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys
3. Use SSH URL: `git@github.com:YOUR_USERNAME/transcription-local-optimized.git`

## Post-Upload Setup

### 1. Update README.md
- Add installation instructions
- Add usage examples
- Add model download instructions

### 2. Add GitHub Actions (Optional)
- Create `.github/workflows/` for CI/CD
- Add Python testing workflow

### 3. Add Issues Template
- Create `.github/ISSUE_TEMPLATE/` for bug reports and feature requests

## Important Notes

1. **Models Directory**: Users will need to download models separately using the provided scripts
2. **Database**: SQLite database is excluded - users will create their own
3. **Environment**: Users need to set up their own environment variables
4. **Audio Files**: Users will upload their own audio files

## Quick Commands Summary

```bash
# Check what will be uploaded
git status

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: Transcription platform"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/transcription-local-optimized.git

# Push
git push -u origin main
```

## Troubleshooting

- If you get authentication errors, use a Personal Access Token
- If files are too large, check `.gitignore` is working properly
- If you need to exclude additional files, update `.gitignore` and run `git rm -r --cached .` then `git add .` 