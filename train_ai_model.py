import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

X = np.load("features.npy")

model = IsolationForest(
    contamination=0.02,
    n_estimators=200,
    max_samples='auto',
    random_state=42
)

model.fit(X)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully")
