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
CHUNK_SIZE = 1024
RATE = 16000

# Load the model
model = whisper.load_model("base")

# Initialize PyAudio
pa = pyaudio.PyAudio()
audio_stream = pa.open(rate=RATE, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=CHUNK_SIZE)

# Initialize temporary wave file
temp_wave_filepath = "temp_wave_file.wav"
temp_wave = wave.open(temp_wave_filepath, "wb")
temp_wave.setnchannels(1)
temp_wave.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
temp_wave.setframerate(RATE)

# Create temporary and permanent log files
temp_log_file = "temp_log.txt"
log_file = f"log_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt"
start_time = time.time()

# Process audio stream
try:
    print("Start speaking...")
    while True:
        audio_data = audio_stream.read(CHUNK_SIZE)
        temp_wave.writeframes(audio_data)

        # Only send data to the model after recording at least 10 seconds of audio
        if temp_wave.getnframes() / RATE >= 10:
            # Close the temporary wave file and send the data to the model
            temp_wave.close()
            audio = whisper.load_audio(temp_wave_filepath)
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # Decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)

            # Print the recognized text and write it to the temporary log file
            print(result.text)
            with open(temp_log_file, "a") as f:
                f.write(f"{result.text}\n")

            # Re-open the temporary wave file and start recording again
            temp_wave = wave.open(temp_wave_filepath, "wb")
            temp_wave.setnchannels(1)
            temp_wave.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            temp_wave.setframerate(RATE)

except KeyboardInterrupt:
    print("\nDone!")

# Clean up
audio_stream.stop_stream()
audio_stream.close()
pa.terminate()
temp_wave.close()
os.remove(temp_wave_filepath)

# Calculate the end time and append the temporary log to the permanent log
end_time = time.time()
rfc3339_start = datetime.fromtimestamp(start_time).isoformat()
rfc3339_end = datetime.fromtimestamp(end_time).isoformat()

with open(log_file, "a") as permanent_log:
    permanent_log.write(f"Start: {rfc3339_start}\n")
    permanent_log.write(f"End: {rfc3339_end}\n\n")
    with open(temp_log_file, "r") as temp_log:
        permanent_log.write(temp_log.read())

# Remove the temporary log file
os.remove(temp_log_file)

