import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import plotly.express as px

# Load dataset
df = pd.read_csv("mehrabad_flights.csv")

# Label Encoding
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df['Airline_enc'] = le_airline.fit_transform(df['Airline'])
df['Destination_enc'] = le_dest.fit_transform(df['Destination'])
df['Weekday_enc'] = le_week.fit_transform(df['Weekday'])

# Model input and output
X = df[['Airline_enc', 'Destination_enc', 'Weekday_enc', 'ScheduledHour']]
y = df['Delayed']

# Train RandomForest
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Prepare scaler for GRU model
scaler = MinMaxScaler()
X_scaled = pd.DataFrame({
    'Airline': le_airline.transform(df['Airline']),
    'Destination': le_dest.transform(df['Destination']),
    'Weekday': le_week.transform(df['Weekday']),
    'ScheduledHour': df['ScheduledHour']
})
scaler.fit(X_scaled)

# Load GRU model
gru_model = load_model("gru_delay_model.h5")

# UI
st.title("پیش‌بینی تأخیر پرواز - مهرآباد")

col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x='ScheduledHour', color='Delayed',
                        title="تعداد پروازها بر اساس ساعت و وضعیت تأخیر")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    delay_avg = df.groupby("Airline")["DelayMinutes"].mean().sort_values()
    fig2 = px.bar(delay_avg, title="میانگین تأخیر هر ایرلاین")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("🔍 فرم پیش‌بینی")

# Input form
airline = st.selectbox("شرکت هواپیمایی", df['Airline'].unique())
destination = st.selectbox("مقصد", df['Destination'].unique())
weekday = st.selectbox("روز هفته", df['Weekday'].unique())
hour = st.slider("ساعت پرواز (۲۴ ساعته)", 0, 23)

# Input for both models
input_data_rf = pd.DataFrame([[
    le_airline.transform([airline])[0],
    le_dest.transform([destination])[0],
    le_week.transform([weekday])[0],
    hour
]], columns=X.columns)

input_data_gru = pd.DataFrame([[
    le_airline.transform([airline])[0],
    le_dest.transform([destination])[0],
    le_week.transform([weekday])[0],
    hour
]], columns=['Airline', 'Destination', 'Weekday', 'ScheduledHour'])

input_scaled_gru = scaler.transform(input_data_gru)
input_reshaped_gru = input_scaled_gru.reshape((1, 1, input_scaled_gru.shape[1]))

# Prediction buttons
col_rf, col_gru = st.columns(2)

with col_rf:
    if st.button("پیش‌بینی با GCN"):
        pred_rf = rf_model.predict(input_data_rf)[0]
        st.info(f"پیش‌بینی GCN: {'🚨 تأخیر دارد' if pred_rf else '✅ بدون تأخیر'}")
        result = pred_rf

with col_gru:
    if st.button("پیش‌بینی با GRU"):
        prediction = gru_model.predict(input_reshaped_gru)[0][0]
        st.write(f"📊 احتمال تأخیر: {round(prediction * 100, 2)}٪")

        result = int(prediction > 0.5)
        if result:
            st.error("🚨 احتمال زیاد تأخیر دارد.")
        else:
            st.success("✅ احتمال زیاد تأخیر ندارد.")

        # Log prediction
        log_df = pd.DataFrame([{
            'DateTime': datetime.datetime.now().isoformat(),
            'Airline': airline,
            'Destination': destination,
            'Weekday': weekday,
            'ScheduledHour': hour,
            'PredictedDelay': prediction
        }])
        log_df.to_csv("predictions_log.csv", mode='a', header=False, index=False)
        st.success("✅ پیش‌بینی ثبت شد.")

# Show past logs
st.subheader("📝 سوابق پیش‌بینی‌شده")
try:
    logs = pd.read_csv("predictions_log.csv", names=[
        "DateTime", "Airline", "Destination", "Weekday", "ScheduledHour", "PredictedDelay"])
    st.dataframe(logs.tail(10))
except FileNotFoundError:
    st.warning("هنوز داده‌ای ثبت نشده است.")
