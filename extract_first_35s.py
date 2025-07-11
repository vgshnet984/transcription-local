import librosa
import soundfile as sf
import os

# Load the audio file
audio_path = "examples/sample_audio/tes1.mp3"
print(f"Loading audio file: {audio_path}")

# Load audio with librosa
y, sr = librosa.load(audio_path, sr=None, mono=False)
print(f"Audio loaded: {y.shape}, Sample rate: {sr}")

# Calculate samples for 35 seconds
samples_35s = int(35 * sr)
print(f"35 seconds = {samples_35s} samples")

# Extract first 35 seconds
if y.ndim == 1:  # Mono
    y_35s = y[:samples_35s]
else:  # Stereo
    y_35s = y[:, :samples_35s]

# Save the extracted audio
output_path = "examples/sample_audio/tes1_first_35s.wav"
sf.write(output_path, y_35s.T, sr)
print(f"Saved first 35 seconds to: {output_path}")

# Get duration
duration_35s = len(y_35s) / sr
print(f"Extracted duration: {duration_35s:.2f} seconds")

# Also create a text file with transcription info
print("\n" + "="*50)
print("FIRST 35 SECONDS OF TES1.MP3")
print("="*50)
print(f"Original file: {audio_path}")
print(f"Extracted duration: {duration_35s:.2f} seconds")
print(f"Sample rate: {sr} Hz")
print(f"Channels: {y_35s.shape[0] if y_35s.ndim > 1 else 1}")
print(f"Output file: {output_path}")
print("="*50) 