import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("mehrabad_flights.csv")

# Encode categorical columns
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Destination"] = le_dest.fit_transform(df["Destination"])
df["Weekday"] = le_week.fit_transform(df["Weekday"])

X = df[["Airline", "Destination", "Weekday", "ScheduledHour"]]
y = df["Delayed"]

# ==========================
# Random Forest
# ==========================

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_rf, y_train_rf)

rf_pred = rf.predict(X_test_rf)

rf_acc = accuracy_score(y_test_rf, rf_pred)

# ==========================
# GRU
# ==========================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_reshaped = X_scaled.reshape(
    (X_scaled.shape[0], 1, X_scaled.shape[1])
)

X_train_gru, X_test_gru, y_train_gru, y_test_gru = train_test_split(
    X_reshaped,
    y,
    test_size=0.2,
    random_state=42
)

model = load_model("gru_delay_model.h5")

loss, gru_acc = model.evaluate(
    X_test_gru,
    y_test_gru,
    verbose=0
)

# ==========================
# Results
# ==========================

print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"GRU Accuracy: {gru_acc:.4f}")

# ==========================
# Visualization
# ==========================

models = ["Random Forest", "GRU"]
scores = [rf_acc, gru_acc]

plt.figure(figsize=(8,5))

bars = plt.bar(models, scores)

plt.title(
    "Model Accuracy Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel("Accuracy")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha="center"
    )

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Comparison chart saved.")