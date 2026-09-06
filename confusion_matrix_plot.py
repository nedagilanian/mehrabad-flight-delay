import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("mehrabad_flights.csv")

# Encode categorical features
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Destination"] = le_dest.fit_transform(df["Destination"])
df["Weekday"] = le_week.fit_transform(df["Weekday"])

# Features
X = df[["Airline", "Destination", "Weekday", "ScheduledHour"]]
y = df["Delayed"]

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for GRU
X_reshaped = X_scaled.reshape(
    (X_scaled.shape[0], 1, X_scaled.shape[1])
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = load_model("gru_delay_model.h5")

# Predict
y_pred_prob = model.predict(X_test)

y_pred = (
    y_pred_prob > 0.5
).astype(int).flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

plt.imshow(cm)

plt.title("GRU Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1], ["No Delay", "Delay"])
plt.yticks([0,1], ["No Delay", "Delay"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved")