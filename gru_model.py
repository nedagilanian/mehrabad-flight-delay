import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split

# داده‌ها را بخوان
df = pd.read_csv("mehrabad_flights.csv")

# تبدیل ویژگی‌های دسته‌ای به عدد
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df['Airline'] = le_airline.fit_transform(df['Airline'])
df['Destination'] = le_dest.fit_transform(df['Destination'])
df['Weekday'] = le_week.fit_transform(df['Weekday'])

# انتخاب ویژگی‌ها و هدف
X = df[['Airline', 'Destination', 'Weekday', 'ScheduledHour']]
y = df['Delayed']

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# تغییر شکل برای ورودی به GRU: [samples, timesteps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# ساخت مدل GRU
model = Sequential()
model.add(GRU(32, input_shape=(1, X_scaled.shape[1]), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# آموزش مدل
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# ذخیره مدل
model.save("gru_delay_model.h5")
print("✅ مدل GRU ذخیره شد.")
