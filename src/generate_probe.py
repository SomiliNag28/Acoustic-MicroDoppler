import numpy as np
import soundfile as sf

# Acoustic Radar Probe Signal
fs = 48000      # Sampling rate (Hz)
duration = 10   # Seconds
f0 = 18000      # Ultrasonic frequency

t = np.arange(0, duration, 1/fs)

# Continuous wave ultrasonic tone
signal = 0.8 * np.sin(2 * np.pi * f0 * t)

# Save to file
sf.write("../signals/probe_signal.wav", signal, fs)

print("Ultrasonic probe saved in signals/probe_signal.wav")