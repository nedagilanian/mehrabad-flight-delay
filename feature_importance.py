import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("mehrabad_flights.csv")

# Encode categorical variables
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Destination"] = le_dest.fit_transform(df["Destination"])
df["Weekday"] = le_week.fit_transform(df["Weekday"])

# Features and target
X = df[["Airline", "Destination", "Weekday", "ScheduledHour"]]
y = df["Delayed"]

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(importance)

# Plot
plt.figure(figsize=(10, 6))

bars = plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title(
    "Feature Importance - Random Forest",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Importance Score")
plt.ylabel("Features")

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center"
    )

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance chart saved.")