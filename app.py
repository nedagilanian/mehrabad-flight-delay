import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.express as px
import base64
from PIL import Image

# ======== تنظیم صفحه ========
st.set_page_config(page_title="پیش‌بینی تأخیر پرواز", layout="wide" , initial_sidebar_state="expanded")

# ======== تعریف تابع ریست ========
def reset():
    st.session_state.step = "form"
    st.session_state.model_type = None
    st.session_state.form_inputs = {}
    st.rerun()

# ======== بارگذاری فونت ========
def load_local_font(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    font_css = f"""
    <style>
    @font-face {{
      font-family: 'BTRAFFIC';
      src: url(data:font/ttf;base64,{encoded}) format('truetype');
    }}
    html, body, .stApp {{
      font-family: 'BTRAFFIC', sans-serif;
      direction: rtl;
      text-align: right;
      background-color: #14213d;
      color: #e5e5e5;
      margin: 0 !important;
      padding: 0 !important;
    }}
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)

load_local_font("BTRAFFIC.TTF")

# ======== تم کلی ========
theming = """
<style>
.block-container {
    padding-top: 2rem !important;   /* فاصله از بالای صفحه */
    padding-left: 1rem !important;
    padding-right: 2rem !important;
}

h1, h2, h3, h4 {
    color: #e5e5e5;
    text-align: right;
    margin-top: 0rem !important;
}

label {
    color: #e5e5e5 !important;
    font-weight: bold;
}

html, body, .stApp {
    font-size: 18px;
    background-color: #14213d !important;
    color: #e5e5e5 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* حذف header و footer و کادرهای اضافی */
header, footer, .main > div:first-child {
    display: none !important;
}

/* حذف نقاط و کادر سفید بالای صفحه */
.stApp > div:first-child {
    margin-top: 0 !important;
}

.stButton>button {
    background-color: #1d3557;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 20px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #457b9d;
}

.stSelectbox>div>div>div {
    color: white !important;
}

.stTextInput>div>div>input,
.stSelectbox>div>div>div,
.stNumberInput>div>div>input {
    background-color: #1d3557;
    color: white;
    border-radius: 6px;
    text-align: right;
}

.stTextInput>div>div>input::placeholder,
.stSelectbox>div>div>div::placeholder,
.stNumberInput>div>div>input::placeholder {
    color: #e5e5e5;
    opacity: 1;
}

.stDataFrame, .stTable {
    background-color: rgba(255,255,255,0.05);
    border-radius: 10px;
}

.sidebar .sidebar-content {
    background-color: #1d3557;
    color: white;
}

img.intro-image {
    width: 100%;
    height: 100px;
    object-fit: cover;
    object-position: top;
    display: block;
    margin: 0 auto 1rem auto;
}

footer {visibility: hidden;}
</style>
"""
st.markdown(theming, unsafe_allow_html=True)

# ======== (ادامه کد شما دقیقا مثل قبل) ========
# بارگذاری داده و مدل‌ها
# ...


# ======== بارگذاری داده و مدل‌ها ========
@st.cache_data
def load_data():
    return pd.read_csv("mehrabad_flights.csv")

@st.cache_resource
def load_models(df):
    le_airline = LabelEncoder()
    le_dest = LabelEncoder()
    le_week = LabelEncoder()

    df['Airline_enc'] = le_airline.fit_transform(df['Airline'])
    df['Destination_enc'] = le_dest.fit_transform(df['Destination'])
    df['Weekday_enc'] = le_week.fit_transform(df['Weekday'])

    X = df[['Airline_enc', 'Destination_enc', 'Weekday_enc', 'ScheduledHour']]
    y = df['Delayed']

    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame({
        'Airline': le_airline.transform(df['Airline']),
        'Destination': le_dest.transform(df['Destination']),
        'Weekday': le_week.transform(df['Weekday']),
        'ScheduledHour': df['ScheduledHour']
    })
    scaler.fit(X_scaled)

    gru_model = load_model("gru_delay_model.h5")

    return rf_model, scaler, gru_model, le_airline, le_dest, le_week

df = load_data()
rf_model, scaler, gru_model, le_airline, le_dest, le_week = load_models(df)

# ======== مقداردهی اولیه استیت ========
if "step" not in st.session_state:
    st.session_state.step = "form"
if "model_type" not in st.session_state:
    st.session_state.model_type = None
if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {}

# ======== سایدبار ========
st.sidebar.image("1.png", width=200)
st.sidebar.title("✈️ منوی برنامه")
page = st.sidebar.radio("برو به صفحه:", ["معرفی", "فرم پیش‌بینی", "سوابق پیش‌بینی", "مقایسه ایرلاین‌ها"])

# ======== صفحات ========

if page == "معرفی":
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    img_base64 = get_base64_of_bin_file('9999.jpg')

    st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    .intro-title {{
        font-size: 40px;
        color: #ffffff;
        text-align: center;
        margin-top: 2rem;
        font-weight: bold;
        text-shadow: 1px 1px 5px #000;
    }}
    .intro-subtitle {{
        font-size: 30px;
        color:  #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 5px #000;
    }}
    .intro-box {{
        background-color: rgba(0, 0, 0, 0.6); 
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        color: #f1f1f1;
        font-size: 18px;
        text-shadow: 1px 1px 3px #000;
    }}
    .footer-info {{
        font-size: 14px;
        color: #111111;
        text-align: center;
        margin-top: auto; /* برای اینکه پایین صفحه بمونه */
        font-style: italic;
        opacity: 0.4;
        padding-bottom: 2rem;
    }}
    </style>

    <div class="intro-title">📌 پیش‌بینی لحظه‌ای جریان ترافیک فرودگاه</div>
    <div class="intro-subtitle">با استفاده از یادگیری عمیق و تحلیل داده</div>

    <div class="intro-box">
        <p>
        این سامانه با استفاده از <strong>مدل‌های یادگیری ماشین AFTPNet (GCN و GRU)</strong><br>
        تأخیر پروازها را بر اساس اطلاعاتی نظیر ایرلاین، مقصد، ساعت و روز هفته پیش‌بینی می‌کند.
        </p>
        <ul>
            <li>پیش‌بینی تأخیر با مدل گراف (GCN)</li>
            <li>پیش‌بینی تأخیر با مدل زمان‌بندی (GRU)</li>
            <li>تحلیل داده‌ها و سوابق پیش‌بینی</li>
        </ul>
    </div>
    <div class="footer-info">
         <strong>شرکت: ایرسا   |  مدیر پروژه: ندا گیلانیان</strong>
    </div>
    """,
    unsafe_allow_html=True
  )




elif page == "فرم پیش‌بینی":

    # مرحله اول: فرم پیش‌بینی
    if st.session_state.step == "form":
        st.markdown("<h2>🛫 فرم پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

        airline = st.selectbox("✈️ شرکت هواپیمایی", df['Airline'].unique())
        destination = st.selectbox("🎯 مقصد پرواز", df['Destination'].unique())
        weekday = st.selectbox("📅 روز هفته", df['Weekday'].unique())
        hour = st.number_input("⏰ ساعت پرواز (۲۴ ساعته)", 0, 23, value=12)

        st.session_state.form_inputs = {
            "airline": airline,
            "destination": destination,
            "weekday": weekday,
            "hour": hour
        }

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ پیش‌بینی با GCN"):
                st.session_state.model_type = "GCN"
                st.session_state.step = "loading"
                st.rerun()

        with col2:
            if st.button("✅ پیش‌بینی با GRU"):
                st.session_state.model_type = "GRU"
                st.session_state.step = "loading"
                st.rerun()

    # مرحله دوم: نمایش gif لودینگ
    elif st.session_state.step == "loading":
        st.markdown("""
            <style>
                .loading-container {
                    height: 85vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
            </style>
        """, unsafe_allow_html=True)

        with open("Flight.gif", "rb") as f:
            gif_data = f.read()
            encoded_gif = base64.b64encode(gif_data).decode()

        st.markdown(f"""
            <div class="loading-container">
                <img src="data:image/gif;base64,{encoded_gif}" style="width:50%; height:auto;" />
            </div>
        """, unsafe_allow_html=True)

        time.sleep(10)  # نمایش gif به مدت 10 ثانیه

        st.session_state.step = "result"
        st.rerun()

    # مرحله سوم: نمایش نتیجه
    elif st.session_state.step == "result":
        st.markdown("<h2>📊 نتیجه پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

        # آماده‌سازی ورودی برای مدل
        airline_enc = le_airline.transform([st.session_state.form_inputs["airline"]])[0]
        destination_enc = le_dest.transform([st.session_state.form_inputs["destination"]])[0]
        weekday_enc = le_week.transform([st.session_state.form_inputs["weekday"]])[0]
        hour_val = st.session_state.form_inputs["hour"]

        if st.session_state.model_type == "GCN":
            X_input = np.array([[airline_enc, destination_enc, weekday_enc, hour_val]])
            pred = rf_model.predict(X_input)[0]
            pred_prob = rf_model.predict_proba(X_input)[0][1]

            st.markdown(f"""
                <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
                🔎 مدل GCN پیش‌بینی کرده است که احتمال تأخیر پرواز: <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
                </div>
                <div style='font-size:20px; font-weight:bold; color:#06d6a0;'>
                ⏳ وضعیت پیش‌بینی شده: {'با تأخیر' if pred == 1 else 'بدون تأخیر'}
                </div>
                """, unsafe_allow_html=True)

        elif st.session_state.model_type == "GRU":
            X_input = np.array([[airline_enc, destination_enc, weekday_enc, hour_val]])
            X_input = X_input.astype(np.float32)
            X_input = np.expand_dims(X_input, axis=0)

            pred_prob = gru_model.predict(X_input)[0][0]
            pred = 1 if pred_prob > 0.5 else 0

            st.markdown(f"""
                <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
                🔎 مدل GRU پیش‌بینی کرده است که احتمال تأخیر پرواز: <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
                </div>
                <div style='font-size:20px; font-weight:bold; color:#06d6a0;'>
                ⏳ وضعیت پیش‌بینی شده: {'با تأخیر' if pred == 1 else 'بدون تأخیر'}
                </div>
                """, unsafe_allow_html=True)

        # ذخیره سوابق پیش‌بینی
        if "history" not in st.session_state:
            st.session_state.history = []

        new_record = {
            "شرکت هواپیمایی": st.session_state.form_inputs["airline"],
            "مقصد": st.session_state.form_inputs["destination"],
            "روز هفته": st.session_state.form_inputs["weekday"],
            "ساعت": st.session_state.form_inputs["hour"],
            "مدل": st.session_state.model_type,
            "احتمال تأخیر": float(pred_prob),
            "نتیجه": "با تأخیر" if pred == 1 else "بدون تأخیر"
        }

        if not st.session_state.get("saved_to_history", False):
            st.session_state.history.append(new_record)
            st.session_state.saved_to_history = True

        # دکمه ریست
        def reset():
            st.session_state.step = "form"
            st.session_state.model_type = None
            st.session_state.form_inputs = {}
            if "saved_to_history" in st.session_state:
                del st.session_state["saved_to_history"]

        st.button("🔄 شروع مجدد", key="restart", on_click=reset)


elif page == "سوابق پیش‌بینی":

    st.markdown("<h2>📋 سوابق پیش‌بینی‌های انجام شده</h2>", unsafe_allow_html=True)

    # فرضا داده‌های ذخیره شده را نمایش می‌دهیم (برای نمونه)
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history)
    else:
        st.write("هیچ سابقه‌ای موجود نیست.")

elif page == "مقایسه ایرلاین‌ها":
    st.markdown("<h2 style='text-align:right;'>📊 مقایسه میانگین تأخیر و تعداد پروازها</h2>", unsafe_allow_html=True)
    avg_delay = df.groupby('Airline')['DelayMinutes'].mean().reset_index(name='AverageDelay')
    flight_counts = df['Airline'].value_counts().reset_index()
    flight_counts.columns = ['Airline', 'FlightCount']
    comparison_df = pd.merge(avg_delay, flight_counts, on='Airline')
    fig = px.bar(comparison_df, x='Airline', y='AverageDelay',
                 color='FlightCount', color_continuous_scale='Blues',
                 labels={'AverageDelay': 'میانگین تأخیر', 'Airline': 'ایرلاین', 'FlightCount': 'تعداد پروازها'},
                 title="مقایسه میانگین تأخیر و تعداد پروازها بر اساس ایرلاین")
    fig.update_layout(xaxis_title="ایرلاین", yaxis_title="میانگین تأخیر (دقیقه)", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

# ======== پایان کد ========