import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import load_model

# خواندن داده
df = pd.read_csv("mehrabad_flights.csv")

# Encoding
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Destination"] = le_dest.fit_transform(df["Destination"])
df["Weekday"] = le_week.fit_transform(df["Weekday"])

# Features & Target
X = df[["Airline", "Destination", "Weekday", "ScheduledHour"]]
y = df["Delayed"]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for GRU
X_reshaped = X_scaled.reshape(
    (X_scaled.shape[0], 1, X_scaled.shape[1])
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped,
    y,
    test_size=0.2,
    random_state=42
)

# Load Model
model = load_model("gru_delay_model.h5")

# Predictions
y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))