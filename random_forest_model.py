import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("mehrabad_flights.csv")

# Encode categorical columns
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Destination"] = le_dest.fit_transform(df["Destination"])
df["Weekday"] = le_week.fit_transform(df["Weekday"])

# Features and target
X = df[["Airline", "Destination", "Weekday", "ScheduledHour"]]
y = df["Delayed"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))