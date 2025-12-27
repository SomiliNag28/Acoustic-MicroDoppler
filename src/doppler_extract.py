import numpy as np
import librosa 
import matplotlib.pyplot as plt
import sys

# Load recorded signals
filename = sys.argv[1]
audio, fs = librosa.load("../data/raw_recordings/" + filename , sr=48000)

# Compute spectrogram
S = librosa.stft(audio, n_fft=4096, hop_length=512)
S_mag = np.abs(S)

# Frequency axis
freqs = librosa.fft_frequencies(sr=fs, n_fft=4096)

# Find index near 18kHz
f0 = 18000
band = np.where((freqs > f0 - 300) & (freqs < f0 + 300))[0]

# Extract doppler band around 18 kHz
doppler_band = S_mag[band, :]

# Clutter cancellation (remove stationary reflections)
mean_spectrum = np.mean(doppler_band, axis=1, keepdims=True)
doppler_motion = doppler_band - mean_spectrum
doppler_motion[doppler_motion < 0] = 0

# Save motion-only Doppler
np.save("../data/processed/" + filename.replace(".wav", "_doppler.npy"), doppler_motion)

# Plot micro-Doppler
plt.imshow(20*np.log10(doppler_motion + 1e-10), 
           aspect='auto', 
           origin='lower',
           extent=[0, doppler_motion.shape[1], f0-300, f0+300])
plt.colorbar(label = "dB")
plt.title("Micro-Doppler around 18kHz")
plt.xlabel("Time Frames")
plt.ylabel("Frequency (Hz)")
plt.grid()

plt.savefig("../results/plots/" + filename.replace(".wav", "_doppler.png"))
plt.show()

print("Doppler extracted & saved succesfully!")