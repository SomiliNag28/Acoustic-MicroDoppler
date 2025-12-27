import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
x = np.load("../data/processed/x.npy")
y = np.load("../data/processed/y.npy")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train classifier
clf = SVC(kernel="rbf", gamma="scale")
clf.fit(x_train, y_train)

# Test
pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred)

print("Test Accuracy : ", acc)

# Save model
joblib.dump(clf, "../results/models/activity_classifier.pkl")
print("Model saved to results/models/activity_classifier.pkl")