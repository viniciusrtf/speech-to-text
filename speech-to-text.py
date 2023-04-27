import os
import pyaudio
import io
import wave
import sys
import numpy as np
import whisper
import time
from datetime import datetime

# Constants
RATE = 16000
CHUNK_SIZE = 1024

# Load the model
model = whisper.load_model("small")

# Initialize PyAudio
pa = pyaudio.PyAudio()
audio_stream = pa.open(rate=RATE, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=CHUNK_SIZE)

# Initialize temporary wave file
temp_wave_filepath = "temp_wave_file.wav"
temp_wave = wave.open(temp_wave_filepath, "wb")
temp_wave.setnchannels(1)
temp_wave.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
temp_wave.setframerate(RATE)

# Create log file
log_file = f"log_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt"
start_time = time.time()

# Flag variable to indicate whether to continue recording
continue_recording = True

# Record audio and process it
try:
    print("Start speaking...")
    while continue_recording:
        audio_data = audio_stream.read(CHUNK_SIZE)
        temp_wave.writeframes(audio_data)

except KeyboardInterrupt:
    print("\nStopping recording...")

# Close the temporary wave file and send the data to the model
temp_wave.close()
audio = whisper.load_audio(temp_wave_filepath)
audio = whisper.pad_or_trim(audio)

# Make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
language_code = max(probs, key=probs.get)
print(f"Detected language: {language_code}")

# Transcribe the audio
result = model.transcribe(audio, language=language_code)

# Print the recognized text
print(result["text"])

# Clean up
audio_stream.stop_stream()
audio_stream.close()
pa.terminate()
os.remove(temp_wave_filepath)

# Calculate the end time and append to the permanent log
end_time = time.time()
rfc3339_start = datetime.fromtimestamp(start_time).isoformat()
rfc3339_end = datetime.fromtimestamp(end_time).isoformat()

with open(log_file, "a") as permanent_log:
    permanent_log.write(f"Start: {rfc3339_start}\n")
    permanent_log.write(f"End: {rfc3339_end}\n\n")
    permanent_log.write(f"Transcription:\n{result['text']}\n\n")

