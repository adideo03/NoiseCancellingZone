import pyaudio

import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# Constants for audio settings
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (samples per second)
CHUNK_SIZE = 1024  # Size of each audio chunk (number of frames)

# Initialize the audio stream
audio = pyaudio.PyAudio()

# Open a streaming stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

try:
    print("Recording... (Press Ctrl+C to stop)")
    while True:
        # Read audio data from the microphone
        audio_data = stream.read(CHUNK_SIZE)

        # Convert audio data to a NumPy array
        numerical_data = np.frombuffer(audio_data, dtype=np.int16)
        print(numerical_data)

        # Process the numerical data as needed (e.g., save it, analyze it, etc.)

except KeyboardInterrupt:
    print("Recording stopped.")

# Close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()


#graph

