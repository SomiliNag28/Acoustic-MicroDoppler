import numpy as np
import os

x = []
y = []

files = os.listdir("../data/processed")
for f in files:
    if f.endswith("_feat.npy"):
        feat = np.load("../data/processed/" + f)
        x.append(feat)

        # Label
        if "empty" in f:
            y.append(0)  # No human
        else:
            y.append(1)  # Human presence

x = np.array(x)
y = np.array(y)

np.save("../data/processed/x.npy", x)
np.save("../data/processed/y.npy", y)

print("Dataset built")
print("x shape :", x.shape)
print("y :", y)