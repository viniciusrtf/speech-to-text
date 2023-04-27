import os
import pyaudio
import io
import wave
import sys
import numpy as np
import whisper

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

# Process audio stream
try:
    print("Start speaking...")
    while True:
        audio_data = audio_stream.read(CHUNK_SIZE)
        temp_wave.writeframes(audio_data)

        # Only send data to the model after recording at least 5 seconds of audio
        if temp_wave.getnframes() / RATE >= 5:
            # Close the temporary wave file and send the data to the model
            temp_wave.close()
            audio = whisper.load_audio(temp_wave_filepath)
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # Decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)

            # Print the recognized text
            sys.stdout.write(f"\r{result.text}")
            sys.stdout.flush()

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

