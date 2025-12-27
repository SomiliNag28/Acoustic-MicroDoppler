import numpy as np
import sys

doppler = np.load("../data/processed/" + sys.argv[1])

# Total motion energy
energy = np.sum(doppler)

# Doppler bandwidth
freq_profile = np.sum(doppler, axis=1)
bandwidth = np.sum(freq_profile > 0.1 * np.max(freq_profile))

# Temporal activity
time_profile = np.sum(doppler, axis=0)
activity = np.sum(time_profile > 0.1 * np.max(time_profile))

# Velocity variance
velocity_var = np.var(freq_profile)

features = np.array([energy, bandwidth, activity, velocity_var])

np.save("../data/processed/" + sys.argv[1].replace("_doppler.npy", "_feat.npy"), features)

print("Features:", features)