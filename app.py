import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



# Load dataset
df = pd.read_csv("mehrabad_flights.csv")


model = load_model("gru_delay_model.h5")

from sklearn.preprocessing import LabelEncoder

le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

# آموزش مجدد روی داده کامل
le_airline.fit(df['Airline'])
le_dest.fit(df['Destination'])
le_week.fit(df['Weekday'])


# آماده‌سازی scaler برای مقیاس‌دهی ورودی جدید
scaler = MinMaxScaler()
X_all = pd.DataFrame({
    'Airline': le_airline.transform(df['Airline']),
    'Destination': le_dest.transform(df['Destination']),
    'Weekday': le_week.transform(df['Weekday']),
    'ScheduledHour': df['ScheduledHour']
})

scaler.fit(X_all)


# Encode categorical columns
le_airline = LabelEncoder()
le_dest = LabelEncoder()
le_week = LabelEncoder()

df['Airline_enc'] = le_airline.fit_transform(df['Airline'])
df['Destination_enc'] = le_dest.fit_transform(df['Destination'])
df['Weekday_enc'] = le_week.fit_transform(df['Weekday'])

X = df[['Airline_enc', 'Destination_enc', 'Weekday_enc', 'ScheduledHour']]
y = df['Delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI
st.title("پیش‌بینی تأخیر پرواز - مهرآباد")
st.subheader("آمار کلی پروازها")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='ScheduledHour', color='Delayed',
                        title="تعداد پروازها بر اساس ساعت و وضعیت تأخیر")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    delay_avg = df.groupby("Airline")["DelayMinutes"].mean().sort_values()
    fig2 = px.bar(delay_avg, title="میانگین تأخیر هر ایرلاین")
    st.plotly_chart(fig2, use_container_width=True)

st.write("با استفاده از اطلاعات پرواز، پیش‌بینی می‌کنیم آیا تأخیر دارد یا خیر.")

airline = st.selectbox("شرکت هواپیمایی", df['Airline'].unique())
destination = st.selectbox("مقصد", df['Destination'].unique())
weekday = st.selectbox("روز هفته", df['Weekday'].unique())
hour = st.slider("ساعت پرواز (۲۴ ساعته)", 0, 23)

# Prediction
# if st.button("پیش‌بینی"):
#     input_data = pd.DataFrame([[
#         le_airline.transform([airline])[0],
#         le_dest.transform([destination])[0],
#         le_week.transform([weekday])[0],
#         hour
#     ]], columns=X.columns)

#     pred = model.predict(input_data)[0]
#     if pred == 1:
#         st.error("🚨 احتمال زیاد تأخیر دارد.")
#     else:
#         st.success("✅ احتمال زیاد تأخیر ندارد.")

# import datetime

# result = {
#     "datetime": datetime.datetime.now().isoformat(),
#     "Airline": airline,
#     "Destination": destination,
#     "Weekday": weekday,
#     "ScheduledHour": hour,
#     "PredictedDelay": int(pred)
# }

if st.button("پیش‌بینی با GRU"):
    input_data = pd.DataFrame([[
        le_airline.transform([airline])[0],
        le_dest.transform([destination])[0],
        le_week.transform([weekday])[0],
        hour
    ]], columns=['Airline', 'Destination', 'Weekday', 'ScheduledHour'])

    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

    prediction = model.predict(input_reshaped)[0][0]
    st.write(f"احتمال تأخیر: {round(prediction * 100, 2)}٪")

    if prediction > 0.5:
        st.error("🚨 احتمال زیاد تأخیر دارد.")
    else:
        st.success("✅ احتمال زیاد تأخیر ندارد.")


# Append to CSV log
predict = st.button('پیش‌بینی با GRU', key='gru_predict')
if predict:
    input_data = pd.DataFrame([[
        le_airline.transform([airline])[0],
        le_dest.transform([destination])[0],
        le_week.transform([weekday])[0],
        hour
    ]], columns=['Airline', 'Destination', 'Weekday', 'ScheduledHour'])

    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

    prediction = model.predict(input_reshaped)[0][0]
    result = prediction  # مقداردهی result

    # نمایش نتیجه
    if result > 0.5:
        st.error(f"⏰ احتمال تأخیر: {result*100:.1f}٪")
    else:
        st.success(f"🛫 پرواز احتمالاً به موقع خواهد بود ({(1-result)*100:.1f}٪)")

    # لاگ کردن نتیجه فقط بعد از پیش‌بینی:
    import datetime
    log_df = pd.DataFrame([{
        'DateTime': datetime.datetime.now().isoformat(),
        'Airline': airline,
        'Destination': destination,
        'Weekday': weekday,
        'ScheduledHour': hour,
        'PredictedDelay': result
    }])
    log_df.to_csv("predictions_log.csv", mode='a', header=False, index=False)

    st.info("✅ نتیجه ثبت شد در log.")


st.subheader("📄 سوابق پیش‌بینی‌شده")

try:
    logs = pd.read_csv("predictions_log.csv", names=[
        "DateTime", "Airline", "Destination", "Weekday", "ScheduledHour", "PredictedDelay"])
    st.dataframe(logs.tail(10))
except FileNotFoundError:
    st.warning("هنوز داده‌ای ذخیره نشده.")
