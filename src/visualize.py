import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load doppler map
doppler = np.load("../data/processed/doppler.npy")

plt.figure(figsize=(10,4))
plt.imshow(20*np.log10(doppler + 1e-10), aspect='auto', origin='lower')
plt.title("Human Micro-Doppler (Motion Only)")
plt.xlabel("Time")
plt.ylabel("Doppler Frequency Bin")
plt.colorbar(label="Energy(dB)")
plt.savefig("../results/plots/micro_doppler.png")
plt.show()

# Load ML model
clf = joblib.load("../results/models/activity_classifier.pkl")

# Load one sample feature
feat = np.load("../data/processed/human1_feat.npy")

pred = clf.predict([feat])[0]

print("Prediction : ", "Human Present" if pred==1 else "Empty Room")