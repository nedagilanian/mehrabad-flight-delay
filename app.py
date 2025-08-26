# # ======== FIRST MVP  ========
# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import time
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from tensorflow.keras.models import load_model
# import plotly.express as px
# import base64
# from PIL import Image
# import plotly.graph_objects as go


# # ======== تنظیم صفحه ========
# st.set_page_config(page_title="پیش‌بینی تأخیر پرواز", layout="wide" , initial_sidebar_state="expanded")

# # ======== تعریف تابع ریست ========
# def reset():
#     st.session_state.step = "form"
#     st.session_state.model_type = None
#     st.session_state.form_inputs = {}
#     st.rerun()

# # ======== بارگذاری فونت ========
# def load_local_font(path):
#     with open(path, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode()
#     font_css = f"""
#     <style>
#     @font-face {{
#       font-family: 'BTRAFFIC';
#       src: url(data:font/ttf;base64,{encoded}) format('truetype');
#     }}
#     html, body, .stApp {{
#       font-family: 'BTRAFFIC', sans-serif;
#       direction: rtl;
#       text-align: right;
#       background-color: #14213d;
#       color: #e5e5e5;
#       margin: 0 !important;
#       padding: 0 !important;
#     }}
#     </style>
#     """
#     st.markdown(font_css, unsafe_allow_html=True)

# load_local_font("BTRAFFIC.TTF")
# ##################
# def show_dashboard_charts(df):
#     avg_delay = df["DelayMinutes"].mean()
#     total_flights = len(df)
#     delayed_ratio = (df["Delayed"].sum() / total_flights) * 100

#     # ===== گیج‌های کوچک =====
#     col1, col2, col3 = st.columns(3)
#     gauges = [
#         {
#             "value": avg_delay,
#             "title": "میانگین تأخیر",
#             "suffix": " دقیقه",
#             "range": [0, 60],
#             "steps": [(0, 15, "#6ee7b7"), (15, 30, "#fde68a"), (30, 60, "#fca5a5")]
#         },
#         {
#             "value": delayed_ratio,
#             "title": "درصد تأخیر",
#             "suffix": "%",
#             "range": [0, 100],
#             "steps": [(0, 20, "#6ee7b7"), (20, 50, "#fde68a"), (50, 100, "#fca5a5")]
#         },
#         {
#             "value": total_flights,
#             "title": "کل پروازها",
#             "suffix": "",
#             "range": [0, total_flights*1.2],
#             "steps": [
#                 (0, total_flights/3, "#6ee7b7"),
#                 (total_flights/3, total_flights*0.66, "#fde68a"),
#                 (total_flights*0.66, total_flights*1.2, "#fdba74")
#             ]
#         }
#     ]

#     for col, g in zip((col1, col2, col3), gauges):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=round(g["value"], 1),
#             number={'suffix': g["suffix"], 'font': {'size': 28}},
#             title={'text': g["title"], 'font': {'size': 16}},
#             gauge={
#                 'axis': {'range': g["range"], 'tickwidth': 0, 'tickcolor': "white"},
#                 'bar': {'color': "#1f2937"},
#                 'steps': [{'range': [s[0], s[1]], 'color': s[2]} for s in g["steps"]],
#                 'borderwidth': 0
#             }
#         ))
#         fig.update_layout(height=180, margin=dict(t=20, b=10, l=10, r=10))
#         col.plotly_chart(fig, use_container_width=True)

#     # ===== نمودارها =====
#     col4, col5 = st.columns(2)
#     airline_delay = df.groupby("Airline")["DelayMinutes"].mean().reset_index()
#     fig_airline = px.bar(
#         airline_delay, x="Airline", y="DelayMinutes",
#         title="میانگین تأخیر بر اساس ایرلاین",
#         color="DelayMinutes",
#         color_continuous_scale=["#6ee7b7", "#fde68a", "#fca5a5"]
#     )
#     fig_airline.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col4.plotly_chart(fig_airline, use_container_width=True)

#     weekday_count = df["Weekday"].value_counts().reset_index()
#     weekday_count.columns = ["Weekday", "Count"]
#     fig_weekday = px.bar(
#         weekday_count, x="Weekday", y="Count",
#         title="تعداد پروازها بر اساس روز هفته",
#         color="Count",
#         color_continuous_scale="Blues"
#     )
#     fig_weekday.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col5.plotly_chart(fig_weekday, use_container_width=True)

#     col6, col7 = st.columns(2)
#     hour_count = df["ScheduledHour"].value_counts().reset_index()
#     hour_count.columns = ["Hour", "Count"]
#     fig_hour = px.bar(
#         hour_count.sort_values("Hour"), x="Hour", y="Count",
#         title="توزیع پروازها بر اساس ساعت",
#         color="Count", color_continuous_scale="Viridis"
#     )
#     fig_hour.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col6.plotly_chart(fig_hour, use_container_width=True)

#     delay_by_hour = df.groupby("ScheduledHour")["DelayMinutes"].mean().reset_index()
#     fig_delay_hour = px.line(
#         delay_by_hour, x="ScheduledHour", y="DelayMinutes",
#         title="میانگین تأخیر بر اساس ساعت پرواز",
#         markers=True
#     )
#     fig_delay_hour.update_traces(line_color="#1f2937")
#     fig_delay_hour.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col7.plotly_chart(fig_delay_hour, use_container_width=True)
# # ======== تم کلی ========
# theming = """
# <style>
# .block-container {
#     padding-top: 2rem !important;   /* فاصله از بالای صفحه */
#     padding-left: 1rem !important;
#     padding-right: 2rem !important;
# }

# h1, h2, h3, h4 {
#     color: #e5e5e5;
#     text-align: right;
#     margin-top: 0rem !important;
# }

# label {
#     color: #e5e5e5 !important;
#     font-weight: bold;
# }

# html, body, .stApp {
#     font-size: 18px;
#     background-color: #14213d !important;
#     color: #e5e5e5 !important;
#     margin: 0 !important;
#     padding: 0 !important;
# }

# /* حذف header و footer و کادرهای اضافی */
# header, footer, .main > div:first-child {
#     display: none !important;
# }

# /* حذف نقاط و کادر سفید بالای صفحه */
# .stApp > div:first-child {
#     margin-top: 0 !important;
# }

# .stButton>button {
#     background-color: #1d3557;
#     color: white;
#     border-radius: 8px;
#     border: none;
#     padding: 8px 20px;
#     font-size: 16px;
# }

# .stButton>button:hover {
#     background-color: #457b9d;
# }

# .stSelectbox>div>div>div {
#     color: white !important;
# }

# .stTextInput>div>div>input,
# .stSelectbox>div>div>div,
# .stNumberInput>div>div>input {
#     background-color: #1d3557;
#     color: white;
#     border-radius: 6px;
#     text-align: right;
# }

# .stTextInput>div>div>input::placeholder,
# .stSelectbox>div>div>div::placeholder,
# .stNumberInput>div>div>input::placeholder {
#     color: #e5e5e5;
#     opacity: 1;
# }

# .stDataFrame, .stTable {
#     background-color: rgba(255,255,255,0.05);
#     border-radius: 10px;
# }

# .sidebar .sidebar-content {
#     background-color: #1d3557;
#     color: white;
# }

# img.intro-image {
#     width: 100%;
#     height: 100px;
#     object-fit: cover;
#     object-position: top;
#     display: block;
#     margin: 0 auto 1rem auto;
# }

# footer {visibility: hidden;}
# </style>
# """
# st.markdown(theming, unsafe_allow_html=True)

# # ======== (ادامه کد شما دقیقا مثل قبل) ========
# # بارگذاری داده و مدل‌ها
# # ...


# # ======== بارگذاری داده و مدل‌ها ========
# @st.cache_data
# def load_data():
#     return pd.read_csv("mehrabad_flights.csv")

# @st.cache_resource
# def load_models(df):
#     le_airline = LabelEncoder()
#     le_dest = LabelEncoder()
#     le_week = LabelEncoder()

#     df['Airline_enc'] = le_airline.fit_transform(df['Airline'])
#     df['Destination_enc'] = le_dest.fit_transform(df['Destination'])
#     df['Weekday_enc'] = le_week.fit_transform(df['Weekday'])

#     X = df[['Airline_enc', 'Destination_enc', 'Weekday_enc', 'ScheduledHour']]
#     y = df['Delayed']

#     rf_model = RandomForestClassifier()
#     rf_model.fit(X, y)

#     scaler = MinMaxScaler()
#     X_scaled = pd.DataFrame({
#         'Airline': le_airline.transform(df['Airline']),
#         'Destination': le_dest.transform(df['Destination']),
#         'Weekday': le_week.transform(df['Weekday']),
#         'ScheduledHour': df['ScheduledHour']
#     })
#     scaler.fit(X_scaled)

#     gru_model = load_model("gru_delay_model.h5")

#     return rf_model, scaler, gru_model, le_airline, le_dest, le_week

# df = load_data()
# rf_model, scaler, gru_model, le_airline, le_dest, le_week = load_models(df)

# # ======== مقداردهی اولیه استیت ========
# if "step" not in st.session_state:
#     st.session_state.step = "form"
# if "model_type" not in st.session_state:
#     st.session_state.model_type = None
# if "form_inputs" not in st.session_state:
#     st.session_state.form_inputs = {}

# # ======== سایدبار ========
# st.sidebar.image("1.png", width=200)
# st.sidebar.title("✈️ ")
# page = st.sidebar.radio("برو به صفحه:", ["معرفی", "شروع پیش‌بینی", "تاریخجه پیش‌بینی", "مقایسه ایرلاین‌ها"])

# # ======== صفحات ========

# if page == "معرفی":
#     def get_base64_of_bin_file(bin_file):
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()

#     img_base64 = get_base64_of_bin_file('133.jpg')

#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#         background-size: cover;
#     }}
#     .intro-title {{
#         font-size: 40px;
#         color: #ffffff;
#         text-align: center;
#         margin-top: 2rem;
#         font-weight: bold;
#         text-shadow: 1px 1px 5px #000;
#     }}
#     .intro-subtitle {{
#         font-size: 30px;
#         color:  #ffffff;
#         text-align: center;
#         margin-bottom: 1rem;
#         text-shadow: 1px 1px 5px #000;
#     }}
#     .intro-box {{
#         background-color: rgba(0, 0, 0, 0.6); 
#         padding: 2rem;
#         border-radius: 12px;
#         margin-top: 2rem;
#         color: #f1f1f1;
#         font-size: 18px;
#         text-shadow: 1px 1px 3px #000;
#     }}
#     .footer-info {{
#         font-size: 15px;
#         color: #ffffff;
#         text-align: center;
#         font-style: italic;
#         opacity: 0.8;
#         margin: 0;
#     }}
#     </style>

#     <div class="intro-title">✈️ پیش‌بینی لحظه‌ای جریان ترافیک فرودگاه</div>
#     <div class="intro-subtitle">با استفاده از یادگیری عمیق و تحلیل داده</div>

#     <div class="intro-box">
#         <p>
#         این سامانه با استفاده از <strong>مدل‌های یادگیری ماشین ATFPNet (GCN و GRU)</strong><br>
#         تأخیر پروازها را بر اساس اطلاعاتی نظیر ایرلاین، مقصد، ساعت و روز هفته پیش‌بینی می‌کند.
#         </p>
#         <ul>
#             <li>پیش‌بینی تأخیر با مدل گراف (GCN)</li>
#             <li>پیش‌بینی تأخیر با مدل زمان‌بندی (GRU)</li>
#             <li>تحلیل داده‌ها و سوابق پیش‌بینی</li>
#         </ul>
#     </div>
#     <div style="position: relative; width: 100%; margin-top: 3rem;">
#     <div style="position: absolute; bottom: 0; width: 100%; background-color: rgba(0,0,0,0.6); padding: 1rem 0;">
#         <p class="footer-info">
#             <strong>شرکت: ایرسا | مدیر پروژه: ندا گیلانیان</strong>
#         </p>
#     </div>
#     </div>

#     """,
#     unsafe_allow_html=True
#   )




# elif page == "شروع پیش‌بینی":

#     def set_background_image(image_path):
#         with open(image_path, "rb") as f:
#             data = f.read()
#         img_base64 = base64.b64encode(data).decode()
#         page_bg_img = f"""
#         <style>
#         .stApp {{
#             background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#             background-size: cover;
#             background-attachment: fixed;
#         }}
#         </style>
#         """
#         st.markdown(page_bg_img, unsafe_allow_html=True)


#     set_background_image("12.jpg")

#     # مرحله اول: فرم پیش‌بینی
#     if st.session_state.step == "form":
#         st.markdown("<h2>🛫 فرم پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

#         airline = st.selectbox("✈️  ایرلاین", df['Airline'].unique())
#         destination = st.selectbox("🎯 مقصد ", df['Destination'].unique())
#         weekday = st.selectbox("📅 روز هفته", df['Weekday'].unique())
#         hour = st.number_input("⏰ انتخاب ساعت پرواز (۲۴ ساعته)", 0, 23, value=12)

#         st.session_state.form_inputs = {
#             "airline": airline,
#             "destination": destination,
#             "weekday": weekday,
#             "hour": hour
#         }

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("✅ پیش‌بینی کن با GCN"):
#                 st.session_state.model_type = "GCN"
#                 st.session_state.step = "loading"
#                 st.rerun()

#         with col2:
#             if st.button("✅ پیش‌بینی کن با GRU"):
#                 st.session_state.model_type = "GRU"
#                 st.session_state.step = "loading"
#                 st.rerun()

  
#     # مرحله دوم: نمایش gif لودینگ
#     elif st.session_state.step == "loading":
#         with open("Flight.gif", "rb") as f:
#             gif_data = f.read()
#             encoded_gif = base64.b64encode(gif_data).decode()

#         st.markdown(f"""
#             <style>
#             .loading-container {{
#                 display: flex;
#                 flex-direction: column;
#                 justify-content: center;
#                 align-items: center;
#                 margin-top: 2rem;
#                 margin-bottom: 2rem;
#                 color: #ffd166;
#                 font-size: 22px;
#                 font-weight: bold;
#                 text-align: center;
#                 direction: rtl;
#                 font-family: 'BTRAFFIC', sans-serif;
#                 user-select: none;
#             }}
#             .loading-text {{
#                 margin-bottom: 1rem;
#                 text-shadow: 1px 1px 2px #000;
#             }}
#             .loading-container img {{
#                 width: 300px;
#                 max-width: 80%;
#                 height: auto;
#                 border-radius: 12px;
#                 box-shadow: 0 4px 10px rgba(0,0,0,0.2);
#             }}
#             </style>
#             <div class="loading-container">
#                 <div class="loading-text">در حال پردازش اطلاعات، لطفا منتظر بمانید...</div>
#                 <img src="data:image/gif;base64,{encoded_gif}" />
#             </div>
#         """, unsafe_allow_html=True)

#         time.sleep(10)  # نمایش gif به مدت 10 ثانیه

#         st.session_state.step = "result"
#         st.rerun()

#     # مرحله سوم: نمایش نتیجه
    
#     elif st.session_state.step == "result":
#         def set_background_image(image_path):
#             with open(image_path, "rb") as f:
#                 data = f.read()
#             img_base64 = base64.b64encode(data).decode()
#             page_bg_img = f"""
#             <style>
#             .stApp {{
#                 background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#                 background-size: cover;
#                 background-attachment: fixed;
#             }}
#             </style>
#             """
#             st.markdown(page_bg_img, unsafe_allow_html=True)

#         set_background_image("12.jpg")

        

#         st.markdown("<h2>📊 نتیجه پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

#         # آماده‌سازی ورودی برای مدل
#         airline_enc = le_airline.transform([st.session_state.form_inputs["airline"]])[0]
#         destination_enc = le_dest.transform([st.session_state.form_inputs["destination"]])[0]
#         weekday_enc = le_week.transform([st.session_state.form_inputs["weekday"]])[0]
#         hour_val = st.session_state.form_inputs["hour"]

#         if st.session_state.model_type == "GCN":
#             X_input = np.array([[airline_enc, destination_enc, weekday_enc, hour_val]])
#             pred = rf_model.predict(X_input)[0]
#             pred_prob = rf_model.predict_proba(X_input)[0][1]

#             st.markdown(f"""
#                 <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
#                 🔎 مدل GCN پیش‌بینی کرده است که احتمال تأخیر پرواز: <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
#                 </div>
#                 <div style='font-size:20px; font-weight:bold; color:#06d6a0;'>
#                 ⏳ وضعیت پیش‌بینی شده: {'با تأخیر' if pred == 1 else 'بدون تأخیر'}
#                 </div>
#                 """, unsafe_allow_html=True)

#         elif st.session_state.model_type == "GRU":
#             X_input = np.array([[airline_enc, destination_enc, weekday_enc, hour_val]])
#             X_input = X_input.astype(np.float32)
#             X_input = np.expand_dims(X_input, axis=0)

#             pred_prob = gru_model.predict(X_input)[0][0]
#             pred = 1 if pred_prob > 0.5 else 0

#             st.markdown(f"""
#                 <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
#                 🔎 مدل GRU پیش‌بینی کرده است که احتمال تأخیر پرواز: <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
#                 </div>
#                 <div style='font-size:20px; font-weight:bold; color:#06d6a0;'>
#                 ⏳ وضعیت پیش‌بینی شده: {'با تأخیر' if pred == 1 else 'بدون تأخیر'}
#                 </div>
#                 """, unsafe_allow_html=True)

#         # ذخیره سوابق پیش‌بینی
#         if "history" not in st.session_state:
#             st.session_state.history = []

#         new_record = {
#             "شرکت هواپیمایی": st.session_state.form_inputs["airline"],
#             "مقصد": st.session_state.form_inputs["destination"],
#             "روز هفته": st.session_state.form_inputs["weekday"],
#             "ساعت": st.session_state.form_inputs["hour"],
#             "مدل": st.session_state.model_type,
#             "احتمال تأخیر": float(pred_prob),
#             "نتیجه": "با تأخیر" if pred == 1 else "بدون تأخیر"
#         }

#         if not st.session_state.get("saved_to_history", False):
#             st.session_state.history.append(new_record)
#             st.session_state.saved_to_history = True

#         # دکمه ریست
#         def reset():
#             st.session_state.step = "form"
#             st.session_state.model_type = None
#             st.session_state.form_inputs = {}
#             if "saved_to_history" in st.session_state:
#                 del st.session_state["saved_to_history"]

#         st.button("🔄 شروع مجدد", key="restart", on_click=reset)



# elif page == "تاریخجه پیش‌بینی":
#     def set_background_image(image_path):
#             with open(image_path, "rb") as f:
#                 data = f.read()
#             img_base64 = base64.b64encode(data).decode()
#             page_bg_img = f"""
#             <style>
#             .stApp {{
#                 background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#                 background-size: cover;
#                 background-attachment: fixed;
#             }}
#             </style>
#             """
#             st.markdown(page_bg_img, unsafe_allow_html=True)

#     set_background_image("14.jpg")
#     st.markdown("<h2>📋 سوابق پیش‌بینی‌های انجام شده</h2>", unsafe_allow_html=True)

#     # فرضا داده‌های ذخیره شده را نمایش می‌دهیم (برای نمونه)
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     if st.session_state.history:
#         df_history = pd.DataFrame(st.session_state.history)
#         st.dataframe(df_history)
#     else:
#         st.write("هیچ سابقه‌ای موجود نیست.")

# elif page == "مقایسه ایرلاین‌ها":
#     st.markdown("<h2 style='text-align:right;'>📊 مقایسه میانگین تأخیر و تعداد پروازها</h2>", unsafe_allow_html=True)
#     avg_delay = df.groupby('Airline')['DelayMinutes'].mean().reset_index(name='AverageDelay')
#     flight_counts = df['Airline'].value_counts().reset_index()
#     flight_counts.columns = ['Airline', 'FlightCount']
#     comparison_df = pd.merge(avg_delay, flight_counts, on='Airline')
#     fig = px.bar(comparison_df, x='Airline', y='AverageDelay',
#                  color='FlightCount', color_continuous_scale='Blues',
#                  labels={'AverageDelay': 'میانگین تأخیر', 'Airline': 'ایرلاین', 'FlightCount': 'تعداد پروازها'},
#                  title="مقایسه میانگین تأخیر و تعداد پروازها بر اساس ایرلاین")
#     fig.update_layout(xaxis_title="ایرلاین", yaxis_title="میانگین تأخیر (دقیقه)", title_x=0.5)
#     st.plotly_chart(fig, use_container_width=True)




# # ======== پایان کد ========
# # ======== END FIRST MVP========
# ############################################################################################################


# # ========  SECOND MVP ========

# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime
# import time
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from tensorflow.keras.models import load_model
# import plotly.express as px
# import base64
# from PIL import Image
# import plotly.graph_objects as go
# import plotly.express as px

# # ======== تنظیم صفحه ========
# st.set_page_config(page_title="پیش‌بینی تأخیر پرواز", layout="wide" , initial_sidebar_state="expanded")

# # ======== تعریف تابع ریست ========
# def reset():
#     st.session_state.step = "form"
#     st.session_state.model_type = None
#     st.session_state.form_inputs = {}
#     st.rerun()

# # ======== بارگذاری فونت ========
# import base64
# import streamlit as st

# def load_local_font(path):
#     with open(path, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode()

#     font_css = f"""
#     <style>
#     @font-face {{
#         font-family: 'BTRAFFIC';
#         src: url(data:font/ttf;base64,{encoded});
#     }}
#     html, body, [class*="st-"], .stApp {{
#         font-family: 'BTRAFFIC', sans-serif !important;
#         direction: rtl;
#         text-align: right;
#     }}
#     h1, h2, h3, h4, h5, h6, p, span, label {{
#         font-family: 'BTRAFFIC', sans-serif !important;
#     }}
#     </style>
#     """
#     st.markdown(font_css, unsafe_allow_html=True)

# # فراخوانی تابع
# load_local_font("BTRAFFIC.TTF")

# #________________________
# def show_dashboard_charts(df):
#     avg_delay = df["DelayMinutes"].mean()
#     total_flights = len(df)
#     delayed_ratio = (df["Delayed"].sum() / total_flights) * 100

#     # ===== گیج‌های کوچک =====
#     col1, col2, col3 = st.columns(3)
#     gauges = [
#         {
#             "value": avg_delay,
#             "title": "میانگین تأخیر",
#             "suffix": " دقیقه",
#             "range": [0, 60],
#             "steps": [(0, 15, "#6ee7b7"), (15, 30, "#fde68a"), (30, 60, "#fca5a5")]
#         },
#         {
#             "value": delayed_ratio,
#             "title": "درصد تأخیر",
#             "suffix": "%",
#             "range": [0, 100],
#             "steps": [(0, 20, "#6ee7b7"), (20, 50, "#fde68a"), (50, 100, "#fca5a5")]
#         },
#         {
#             "value": total_flights,
#             "title": "کل پروازها",
#             "suffix": "",
#             "range": [0, total_flights*1.2],
#             "steps": [
#                 (0, total_flights/3, "#6ee7b7"),
#                 (total_flights/3, total_flights*0.66, "#fde68a"),
#                 (total_flights*0.66, total_flights*1.2, "#fdba74")
#             ]
#         }
#     ]

#     for col, g in zip((col1, col2, col3), gauges):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=round(g["value"], 1),
#             number={'suffix': g["suffix"], 'font': {'size': 28}},
#             title={'text': g["title"], 'font': {'size': 16}},
#             gauge={
#                 'axis': {'range': g["range"], 'tickwidth': 0, 'tickcolor': "white"},
#                 'bar': {'color': "#1f2937"},
#                 'steps': [{'range': [s[0], s[1]], 'color': s[2]} for s in g["steps"]],
#                 'borderwidth': 0
#             }
#         ))
#         fig.update_layout(height=180, margin=dict(t=20, b=10, l=10, r=10))
#         col.plotly_chart(fig, use_container_width=True)

#     # ===== نمودارها =====
#     col4, col5 = st.columns(2)
#     airline_delay = df.groupby("Airline")["DelayMinutes"].mean().reset_index()
#     fig_airline = px.bar(
#         airline_delay, x="Airline", y="DelayMinutes",
#         title="میانگین تأخیر بر اساس ایرلاین",
#         color="DelayMinutes",
#         color_continuous_scale=["#6ee7b7", "#fde68a", "#fca5a5"]
#     )
#     fig_airline.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col4.plotly_chart(fig_airline, use_container_width=True)

#     weekday_count = df["Weekday"].value_counts().reset_index()
#     weekday_count.columns = ["Weekday", "Count"]
#     fig_weekday = px.bar(
#         weekday_count, x="Weekday", y="Count",
#         title="تعداد پروازها بر اساس روز هفته",
#         color="Count",
#         color_continuous_scale="Blues"
#     )
#     fig_weekday.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col5.plotly_chart(fig_weekday, use_container_width=True)

#     col6, col7 = st.columns(2)
#     hour_count = df["ScheduledHour"].value_counts().reset_index()
#     hour_count.columns = ["Hour", "Count"]
#     fig_hour = px.bar(
#         hour_count.sort_values("Hour"), x="Hour", y="Count",
#         title="توزیع پروازها بر اساس ساعت",
#         color="Count", color_continuous_scale="Viridis"
#     )
#     fig_hour.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col6.plotly_chart(fig_hour, use_container_width=True)

#     delay_by_hour = df.groupby("ScheduledHour")["DelayMinutes"].mean().reset_index()
#     fig_delay_hour = px.line(
#         delay_by_hour, x="ScheduledHour", y="DelayMinutes",
#         title="میانگین تأخیر بر اساس ساعت پرواز",
#         markers=True
#     )
#     fig_delay_hour.update_traces(line_color="#1f2937")
#     fig_delay_hour.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
#     col7.plotly_chart(fig_delay_hour, use_container_width=True)
# ======== تم کلی ========
# theming = """
# <style>
# .block-container {
#     padding-top: 2rem !important;   /* فاصله از بالای صفحه */
#     padding-left: 1rem !important;
#     padding-right: 2rem !important;
# }

# h1, h2, h3, h4 {
#     color: #e5e5e5;
#     text-align: right;
#     margin-top: 0rem !important;
# }

# label {
#     color: #e5e5e5 !important;
#     font-weight: bold;
# }

# html, body, .stApp {
#     font-size: 18px;
#     background-color: #14213d !important;
#     color: #e5e5e5 !important;
#     margin: 0 !important;
#     padding: 0 !important;
# }

# /* حذف header و footer و کادرهای اضافی */
# header, footer, .main > div:first-child {
#     display: none !important;
# }

# /* حذف نقاط و کادر سفید بالای صفحه */
# .stApp > div:first-child {
#     margin-top: 0 !important;
# }

# .stButton>button {
#     background-color: #1d3557;
#     color: white;
#     border-radius: 8px;
#     border: none;
#     padding: 8px 20px;
#     font-size: 16px;
# }

# .stButton>button:hover {
#     background-color: #457b9d;
# }

# .stSelectbox>div>div>div {
#     color: white !important;
# }

# .stTextInput>div>div>input,
# .stSelectbox>div>div>div,
# .stNumberInput>div>div>input {
#     background-color: #1d3557;
#     color: white;
#     border-radius: 6px;
#     text-align: right;
# }

# .stTextInput>div>div>input::placeholder,
# .stSelectbox>div>div>div::placeholder,
# .stNumberInput>div>div>input::placeholder {
#     color: #e5e5e5;
#     opacity: 1;
# }

# .stDataFrame, .stTable {
#     background-color: rgba(255,255,255,0.05);
#     border-radius: 10px;
# }

# .sidebar .sidebar-content {
#     background-color: #1d3557;
#     color: white;
# }

# img.intro-image {
#     width: 100%;
#     height: 100px;
#     object-fit: cover;
#     object-position: top;
#     display: block;
#     margin: 0 auto 1rem auto;
# }

# .main {background-color: #f7f9fb;}
# .stPlotlyChart {
#     background-color: white;
#     border-radius: 12px;
#     padding: 10px;
#     box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
#     }
#         /* عنوان‌ها */
#     h1, h2, h3 {
#         font-family: 'IRANSans', sans-serif;
#         font-weight: 500;
#     }

# footer {visibility: hidden;}
# </style>
# """
# st.markdown(theming, unsafe_allow_html=True)


# # ======== بارگذاری داده و مدل‌ها ========
# @st.cache_data
# def load_data():
#     return pd.read_csv("mehrabad_flights.csv")

# @st.cache_resource
# def load_models(df):
#     le_airline = LabelEncoder()
#     le_dest = LabelEncoder()
#     le_week = LabelEncoder()

#     df['Airline_enc'] = le_airline.fit_transform(df['Airline'])
#     df['Destination_enc'] = le_dest.fit_transform(df['Destination'])
#     df['Weekday_enc'] = le_week.fit_transform(df['Weekday'])

#     X = df[['Airline_enc', 'Destination_enc', 'Weekday_enc', 'ScheduledHour']]
#     y = df['Delayed']

#     rf_model = RandomForestClassifier()
#     rf_model.fit(X, y)

#     scaler = MinMaxScaler()
#     X_scaled = pd.DataFrame({
#         'Airline': le_airline.transform(df['Airline']),
#         'Destination': le_dest.transform(df['Destination']),
#         'Weekday': le_week.transform(df['Weekday']),
#         'ScheduledHour': df['ScheduledHour']
#     })
#     scaler.fit(X_scaled)

#     gru_model = load_model("gru_delay_model.h5")

#     return rf_model, scaler, gru_model, le_airline, le_dest, le_week

# df = load_data()
# rf_model, scaler, gru_model, le_airline, le_dest, le_week = load_models(df)

# # ======== مقداردهی اولیه استیت ========
# if "step" not in st.session_state:
#     st.session_state.step = "form"
# if "model_type" not in st.session_state:
#     st.session_state.model_type = None
# if "form_inputs" not in st.session_state:
#     st.session_state.form_inputs = {}

# # ======== سایدبار ========
# st.sidebar.image("1.png", width=200)
# st.sidebar.title("✈️ ")
# page = st.sidebar.radio("برو به صفحه:", ["معرفی", "شروع پیش‌بینی", "تاریخجه پیش‌بینی", "تحلیل ایرلاین‌ها", "تحلیل"])

# # ======== صفحات ========

# if page == "معرفی":
#     def get_base64_of_bin_file(bin_file):
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()

#     img_base64 = get_base64_of_bin_file('133.jpg')

#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#         background-size: cover;
#     }}
#     .intro-title {{
#         font-size: 40px;
#         color: #ffffff;
#         text-align: center;
#         margin-top: 2rem;
#         font-weight: bold;
#         text-shadow: 1px 1px 5px #000;
#     }}
#     .intro-subtitle {{
#         font-size: 30px;
#         color:  #ffffff;
#         text-align: center;
#         margin-bottom: 1rem;
#         text-shadow: 1px 1px 5px #000;
#     }}
#     .intro-box {{
#         background-color: rgba(0, 0, 0, 0.6); 
#         padding: 2rem;
#         border-radius: 12px;
#         margin-top: 2rem;
#         color: #f1f1f1;
#         font-size: 18px;
#         text-shadow: 1px 1px 3px #000;
#     }}
#     .footer-info {{
#         font-size: 15px;
#         color: #ffffff;
#         text-align: center;
#         font-style: italic;
#         opacity: 0.8;
#         margin: 0;
#     }}
#     </style>

#     <div class="intro-title">✈️ پیش‌بینی لحظه‌ای جریان ترافیک فرودگاه</div>
#     <div class="intro-subtitle">با استفاده از یادگیری عمیق و تحلیل داده</div>

#     <div class="intro-box">
#         <p>
#         این سامانه با استفاده از <strong>مدل‌های یادگیری ماشین ATFPNet (GCN و GRU)</strong><br>
#         تأخیر پروازها را بر اساس اطلاعاتی نظیر ایرلاین، مقصد، ساعت و روز هفته پیش‌بینی می‌کند.
#         </p>
#         <ul>
#             <li>پیش‌بینی تأخیر با مدل گراف (GCN)</li>
#             <li>پیش‌بینی تأخیر با مدل زمان‌بندی (GRU)</li>
#             <li>تحلیل داده‌ها و سوابق پیش‌بینی</li>
#         </ul>
#     </div>
#     <div style="position: relative; width: 100%; margin-top: 3rem;">
#     <div style="position: absolute; bottom: 0; width: 100%; background-color: rgba(0,0,0,0.6); padding: 1rem 0;">
#         <p class="footer-info">
#             <strong>شرکت: ایرسا | مدیر پروژه: ندا گیلانیان</strong>
#         </p>
#     </div>
#     </div>

#     """,
#     unsafe_allow_html=True
#   )




# elif page == "شروع پیش‌بینی":

#     def set_background_image(image_path):
#         with open(image_path, "rb") as f:
#             data = f.read()
#         img_base64 = base64.b64encode(data).decode()
#         page_bg_img = f"""
#         <style>
#         .stApp {{
#             background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#             background-size: cover;
#             background-attachment: fixed;
#         }}
#         </style>
#         """
#         st.markdown(page_bg_img, unsafe_allow_html=True)


#     set_background_image("12.jpg")

#     # مرحله اول: فرم پیش‌بینی
#     if st.session_state.step == "form":
#         st.markdown("<h2>🛫 فرم پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

#         airline = st.selectbox("✈️  ایرلاین", df['Airline'].unique())
#         destination = st.selectbox("🎯 مقصد ", df['Destination'].unique())
#         weekday = st.selectbox("📅 روز هفته", df['Weekday'].unique())
#         hour = st.number_input("⏰ انتخاب ساعت پرواز (۲۴ ساعته)", 0, 23, value=12)

#         st.session_state.form_inputs = {
#             "airline": airline,
#             "destination": destination,
#             "weekday": weekday,
#             "hour": hour
#         }

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("✅ پیش‌بینی کن با GCN"):
#                 st.session_state.model_type = "GCN"
#                 st.session_state.step = "loading"
#                 st.rerun()

#         with col2:
#             if st.button("✅ پیش‌بینی کن با GRU"):
#                 st.session_state.model_type = "GRU"
#                 st.session_state.step = "loading"
#                 st.rerun()

  
#     # مرحله دوم: نمایش gif لودینگ
#     elif st.session_state.step == "loading":
#         with open("Flight.gif", "rb") as f:
#             gif_data = f.read()
#             encoded_gif = base64.b64encode(gif_data).decode()

#         st.markdown(f"""
#             <style>
#             .loading-container {{
#                 display: flex;
#                 flex-direction: column;
#                 justify-content: center;
#                 align-items: center;
#                 margin-top: 2rem;
#                 margin-bottom: 2rem;
#                 color: #ffd166;
#                 font-size: 22px;
#                 font-weight: bold;
#                 text-align: center;
#                 direction: rtl;
#                 font-family: 'BTRAFFIC', sans-serif;
#                 user-select: none;
#             }}
#             .loading-text {{
#                 margin-bottom: 1rem;
#                 text-shadow: 1px 1px 2px #000;
#             }}
#             .loading-container img {{
#                 width: 300px;
#                 max-width: 80%;
#                 height: auto;
#                 border-radius: 12px;
#                 box-shadow: 0 4px 10px rgba(0,0,0,0.2);
#             }}
#             </style>
#             <div class="loading-container">
#                 <div class="loading-text">در حال پردازش اطلاعات، لطفا منتظر بمانید...</div>
#                 <img src="data:image/gif;base64,{encoded_gif}" />
#             </div>
#         """, unsafe_allow_html=True)

#         time.sleep(10)  # نمایش gif به مدت 10 ثانیه

#         st.session_state.step = "result"
#         st.rerun()

#     # مرحله سوم: نمایش نتیجه
    
#     elif st.session_state.step == "result":
#         def set_background_image(image_path):
#             with open(image_path, "rb") as f:
#                 data = f.read()
#             img_base64 = base64.b64encode(data).decode()
#             page_bg_img = f"""
#             <style>
#             .stApp {{
#                 background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#                 background-size: cover;
#                 background-attachment: fixed;
#             }}
#             </style>
#             """
#             st.markdown(page_bg_img, unsafe_allow_html=True)

#         set_background_image("12.jpg")

        

#         st.markdown("<h2>📊 نتیجه پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

#         # آماده‌سازی ورودی برای مدل
#         airline_enc = le_airline.transform([st.session_state.form_inputs["airline"]])[0]
#         destination_enc = le_dest.transform([st.session_state.form_inputs["destination"]])[0]
#         weekday_enc = le_week.transform([st.session_state.form_inputs["weekday"]])[0]
#         hour_val = st.session_state.form_inputs["hour"]

#         if st.session_state.model_type == "GCN":
#             X_input = np.array([[airline_enc, destination_enc, weekday_enc, hour_val]])
#             pred = rf_model.predict(X_input)[0]
#             pred_prob = rf_model.predict_proba(X_input)[0][1]

#             st.markdown(f"""
#                 <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
#                 🔎 مدل GCN پیش‌بینی کرده است که احتمال تأخیر پرواز: <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
#                 </div>
#                 <div style='font-size:20px; font-weight:bold; color:#06d6a0;'>
#                 ⏳ وضعیت پیش‌بینی شده: {'با تأخیر' if pred == 1 else 'بدون تأخیر'}
#                 </div>
#                 """, unsafe_allow_html=True)

#         elif st.session_state.model_type == "GRU":
#             X_input = np.array([[airline_enc, destination_enc, weekday_enc, hour_val]])
#             X_input = X_input.astype(np.float32)
#             X_input = np.expand_dims(X_input, axis=0)

#             pred_prob = gru_model.predict(X_input)[0][0]
#             pred = 1 if pred_prob > 0.5 else 0

#             st.markdown(f"""
#                 <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
#                 🔎 مدل GRU پیش‌بینی کرده است که احتمال تأخیر پرواز: <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
#                 </div>
#                 <div style='font-size:20px; font-weight:bold; color:#06d6a0;'>
#                 ⏳ وضعیت پیش‌بینی شده: {'با تأخیر' if pred == 1 else 'بدون تأخیر'}
#                 </div>
#                 """, unsafe_allow_html=True)

#         # ذخیره سوابق پیش‌بینی
#         if "history" not in st.session_state:
#             st.session_state.history = []

#         new_record = {
#             "شرکت هواپیمایی": st.session_state.form_inputs["airline"],
#             "مقصد": st.session_state.form_inputs["destination"],
#             "روز هفته": st.session_state.form_inputs["weekday"],
#             "ساعت": st.session_state.form_inputs["hour"],
#             "مدل": st.session_state.model_type,
#             "احتمال تأخیر": float(pred_prob),
#             "نتیجه": "با تأخیر" if pred == 1 else "بدون تأخیر"
#         }

#         if not st.session_state.get("saved_to_history", False):
#             st.session_state.history.append(new_record)
#             st.session_state.saved_to_history = True

#         # دکمه ریست
#         def reset():
#             st.session_state.step = "form"
#             st.session_state.model_type = None
#             st.session_state.form_inputs = {}
#             if "saved_to_history" in st.session_state:
#                 del st.session_state["saved_to_history"]

#         st.button("🔄 شروع مجدد", key="restart", on_click=reset)



# elif page == "تاریخجه پیش‌بینی":
#     def set_background_image(image_path):
#             with open(image_path, "rb") as f:
#                 data = f.read()
#             img_base64 = base64.b64encode(data).decode()
#             page_bg_img = f"""
#             <style>
#             .stApp {{
#                 background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
#                 background-size: cover;
#                 background-attachment: fixed;
#             }}
#             </style>
#             """
#             st.markdown(page_bg_img, unsafe_allow_html=True)
    
#     set_background_image("14.jpg")
#     st.markdown("<h2>📋 سوابق پیش‌بینی‌های انجام شده</h2>", unsafe_allow_html=True)

#     # فرضا داده‌های ذخیره شده را نمایش می‌دهیم (برای نمونه)
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     if st.session_state.history:
#         df_history = pd.DataFrame(st.session_state.history)
#         st.dataframe(df_history)
#     else:
#         st.write("هیچ سابقه‌ای موجود نیست.")

# elif page == "تحلیل ایرلاین‌ها":
    
#     st.markdown("<h2 style='text-align:right;'>📊 مقایسه میانگین تأخیر و تعداد پروازها</h2>", unsafe_allow_html=True)
#     avg_delay = df.groupby('Airline')['DelayMinutes'].mean().reset_index(name='AverageDelay')
#     flight_counts = df['Airline'].value_counts().reset_index()
#     flight_counts.columns = ['Airline', 'FlightCount']
#     comparison_df = pd.merge(avg_delay, flight_counts, on='Airline')
#     fig = px.bar(comparison_df, x='Airline', y='AverageDelay',
#                  color='FlightCount', color_continuous_scale='Blues',
#                  labels={'AverageDelay': 'میانگین تأخیر', 'Airline': 'ایرلاین', 'FlightCount': 'تعداد پروازها'},
#                  title="مقایسه میانگین تأخیر و تعداد پروازها بر اساس ایرلاین")
#     fig.update_layout(xaxis_title="ایرلاین", yaxis_title="میانگین تأخیر (دقیقه)", title_x=0.5)
#     st.plotly_chart(fig, use_container_width=True)

# elif page == "تحلیل":
#     show_dashboard_charts(df)
# # ======== پایان کد ========
# # ========  END SECOND MVP ========
# ############################################################################################################

# ========  THIRD MVP (CLEAN) ========
# 

###############################################
#forth
# ----------------- Imports -----------------
import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from persiantools.jdatetime import JalaliDate


# ----------------- Setup -----------------
st.set_page_config(page_title="پیش‌بینی تأخیر پرواز", layout="wide", initial_sidebar_state="expanded")

# ----------------- Session init -----------------
def ss_init():
    defaults = dict(
        step="form",
        model_type=None,
        form_inputs={},
        data_source=None,   # "CSV" | "API"
        data_path=None,     # UploadedFile | str (URL)
        logged_in=False,
        role=None,          # "admin" | "user"
        current_page="معرفی",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ss_init()

# ----------------- Helpers -----------------
def goto(page_name: str):
    st.session_state.current_page = page_name

def reset_form():
    st.session_state.step = "form"
    st.session_state.model_type = None
    st.session_state.form_inputs = {}
    if "saved_to_history" in st.session_state:
        del st.session_state["saved_to_history"]

# ----------------- Font & Theme -----------------
def load_local_font(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
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
    }}
    </style>
    """, unsafe_allow_html=True)

load_local_font("BTRAFFIC.TTF")

st.markdown("""
<style>
.block-container { padding-top: 2rem !important; padding-left: 1rem !important; padding-right: 2rem !important; }
h1, h2, h3, h4 { color: #e5e5e5; text-align: right; margin-top: 0rem !important; }
label { color: #e5e5e5 !important; font-weight: bold; }
.stButton>button { background-color: #1d3557; color: white; border-radius: 8px; border: none; padding: 8px 20px; font-size: 16px; }
.stButton>button:hover { background-color: #457b9d; }
.stTextInput>div>div>input, .stSelectbox>div>div>div { background-color: #1d3557; color: white; border-radius: 6px; text-align: right; }
.stDataFrame, .stTable { background-color: rgba(255,255,255,0.05); border-radius: 10px; }
.sidebar .sidebar-content { background-color: #1d3557; color: white; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# # ======== تم کلی ========
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


if "current_page" not in st.session_state:
    st.session_state.current_page = "معرفی"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
# ----------------- Sidebar -----------------
st.sidebar.image("1.png", width=200)
st.sidebar.title("✈️ سیستم پیش‌بینی")

# نمایش اسم کاربر یا نقش بالای سایدبار بعد از ورود
if st.session_state.logged_in:
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; font-size:14px; margin:6px 0; 
        padding:4px; border-radius:6px; background-color:#f0f2f6;">
        👤 <b>{st.session_state.role}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# دکمه‌های ورود/خروج و تنظیمات
col_a, col_b = st.sidebar.columns([1,1])

with col_a:
    if not st.session_state.logged_in:
        if st.sidebar.button("🔑 ورود", use_container_width=True):
            goto("login")
    else:
        if st.sidebar.button("🚪 خروج", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.role = None
            goto("معرفی")

# with col_b:
#     if st.sidebar.button("⚙️ تنظیمات", use_container_width=True):
#         if st.session_state.logged_in and st.session_state.role == "admin":
#             st.session_state.current_page = "تنظیمات ادمین"
#         # else:
        #     st.sidebar.warning("برای ورود به تنظیمات باید ادمین باشید.")

# ----------------- منوی صفحات -----------------
menu_pages = ["معرفی", "فرم پیش‌بینی", "سوابق پیش‌بینی", "مقایسه ایرلاین‌ها", "داشبورد تحلیل پروازها","درباره ما"]
if st.session_state.logged_in and st.session_state.role == "admin":
    menu_pages.append("تنظیمات ادمین")

if st.session_state.current_page != "login":
    try:
        selected = st.sidebar.radio("صفحات:", menu_pages, index=menu_pages.index(st.session_state.current_page))
    except ValueError:
        selected = "معرفی"
    st.session_state.current_page = selected


# ----------------- تبدیل ستون‌های فارسی به انگلیسی -----------------
def preprocess_columns(df):
    df = df.copy()
    rename_map = {
        "پرواز": "Airline",
        "مبدا": "Origin",
        "مقصد": "Destination",
        "تاخیر (دقیقه)": "Delay"
    }
    df = df.rename(columns=rename_map)
    
    # افزودن ستون Weekday برای نمونه (اختیاری)
    if 'Weekday' not in df.columns:
        df['Weekday'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][:len(df)]
    
    return df


def load_csv_sample_data():
    return pd.DataFrame({
        "پرواز": ["IR655", "W500", "IR220", "A312", "W502"],
        "مبدا": ["تهران", "مشهد", "اصفهان", "شیراز", "تبریز"],
        "مقصد": ["مشهد", "تهران", "شیراز", "تبریز", "اصفهان"],
        "تاخیر (دقیقه)": [15, 0, 45, 10, 5]
    })

def load_api_sample_data():
    return pd.DataFrame({
        "پرواز": ["API101", "API102", "API103", "API104"],
        "مبدا": ["تهران", "اصفهان", "مشهد", "شیراز"],
        "مقصد": ["تبریز", "شیراز", "تهران", "مشهد"],
        "تاخیر (دقیقه)": [5, 12, 0, 8]
    })

# def get_data(data_source, data_path):
#     if data_source == "CSV":
#         st.info("استفاده از داده  CSV")
#         return load_csv_sample_data()
#     elif data_source == "API":
#         st.info(f"استفاده از داده  API")
#         return load_api_sample_data()
# else:
#     return pd.DataFrame()

def build_models(df_in):
    df = df_in.copy()
    df.columns = df.columns.str.strip()

    le_airline = LabelEncoder()
    le_dest = LabelEncoder()
    le_week = LabelEncoder()

    if 'Airline' in df.columns:
        df['Airline_enc'] = le_airline.fit_transform(df['Airline'])
    if 'Destination' in df.columns:
        df['Destination_enc'] = le_dest.fit_transform(df['Destination'])
    if 'Weekday' in df.columns:
        df['Weekday_enc'] = le_week.fit_transform(df['Weekday'])

    scaler = StandardScaler()
    features = [c for c in ['Airline_enc', 'Destination_enc', 'Weekday_enc'] if c in df.columns]
    X = scaler.fit_transform(df[features]) if features else np.zeros((len(df),1))
    y = df['Delay'] if 'Delay' in df.columns else np.zeros(len(df))

    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    return rf_model, scaler, le_airline, le_dest, le_week

# ----------------- Intro Page -----------------
def intro_page():
    try:
        with open('133.jpg', 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .intro-title {{ font-size: 40px; color: #ffffff; text-align: center; margin-top: 2rem; font-weight: bold; text-shadow: 1px 1px 5px #000; }}
        .intro-subtitle {{ font-size: 30px; color: #ffffff; text-align: center; margin-bottom: 1rem; text-shadow: 1px 1px 5px #000; }}
        .intro-box {{ background-color: rgba(0, 0, 0, 0.6); padding: 2rem; border-radius: 12px; margin-top: 2rem; color: #f1f1f1; font-size: 18px; text-shadow: 1px 1px 3px #000; }}
        .intro-link {{ color: #ffd166; font-weight: bold; text-decoration: underline; cursor: pointer; }}
        </style>
        <div class="intro-title">✈️ پیش‌بینی لحظه‌ای جریان ترافیک فرودگاه</div>
        <div class="intro-subtitle">با استفاده از یادگیری عمیق و تحلیل داده</div>

        <div class="intro-box">
            <p>
            این سامانه با استفاده از <strong>مدل‌های یادگیری ماشین ATFPNet (GCN و GRU)</strong><br>
            تأخیر پروازها را بر اساس اطلاعاتی نظیر ایرلاین، مقصد، ساعت و روز هفته پیش‌بینی می‌کند.
            </p>
            <ul>
                <li>پیش‌بینی تأخیر با مدل گراف (GCN)</li>
                <li>پیش‌بینی تأخیر با مدل زمان‌بندی (GRU)</li>
                <li>تحلیل داده‌ها و سوابق پیش‌بینی</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("تصویر intro موجود نیست!")

if st.session_state.current_page == "معرفی":
   intro_page()

# ----------------- Prediction Page -----------------
def prediction_page():
    import base64
    import time

    # پس‌زمینه
    def set_background_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        img_base64 = base64.b64encode(data).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background_image("12.jpg")

    # داده‌های تستی ساده
    airlines = ["IranAir", "Mahan", "Qeshm", "Aseman"]
    destinations = ["Tehran", "Mashhad", "Shiraz", "Tabriz"]
    weekdays = ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]

    # مرحله اول: فرم
    if st.session_state.step == "form":
        st.markdown("<h2>🛫 فرم پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

        airline = st.selectbox("✈️  ایرلاین", airlines)
        destination = st.selectbox("🎯 مقصد ", destinations)
        weekday = st.selectbox("📅 روز هفته", weekdays)
        hour = st.number_input("⏰ انتخاب ساعت پرواز (۲۴ ساعته)", 0, 23, value=12)

        st.session_state.form_inputs = {
            "airline": airline,
            "destination": destination,
            "weekday": weekday,
            "hour": hour
        }

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ پیش‌بینی کن با GCN"):
                st.session_state.model_type = "GCN"
                st.session_state.step = "loading"
                st.rerun()

        with col2:
            if st.button("✅ پیش‌بینی کن با GRU"):
                st.session_state.model_type = "GRU"
                st.session_state.step = "loading"
                st.rerun()

    # مرحله دوم: نمایش gif لودینگ
    elif st.session_state.step == "loading":
        with open("Flight.gif", "rb") as f:
            gif_data = f.read()
            encoded_gif = base64.b64encode(gif_data).decode()

        st.markdown(f"""
            <style>
            .loading-container {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                margin-top: 2rem;
                margin-bottom: 2rem;
                color: #ffd166;
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                direction: rtl;
                font-family: 'BTRAFFIC', sans-serif;
                user-select: none;
            }}
            .loading-text {{
                margin-bottom: 1rem;
                text-shadow: 1px 1px 2px #000;
            }}
            .loading-container img {{
                width: 300px;
                max-width: 80%;
                height: auto;
                border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            }}
            </style>
            <div class="loading-container">
                <div class="loading-text">در حال پردازش اطلاعات، لطفا منتظر بمانید...</div>
                <img src="data:image/gif;base64,{encoded_gif}" />
            </div>
        """, unsafe_allow_html=True)

        time.sleep(10)  # نمایش gif به مدت 10 ثانیه

        st.session_state.step = "result"
        st.rerun()

    # مرحله سوم: نمایش نتیجه
    
    elif st.session_state.step == "result":
        def set_background_image(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            img_base64 = base64.b64encode(data).decode()
            page_bg_img = f"""
            <style>
            .stApp {{
                background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
            """
            st.markdown(page_bg_img, unsafe_allow_html=True)

        set_background_image("12.jpg")

        

        st.markdown("<h2>📊 نتیجه پیش‌بینی تأخیر پرواز</h2>", unsafe_allow_html=True)

        import random
        pred_prob = random.random()
        pred = 1 if pred_prob > 0.5 else 0

        st.markdown(f"""
            <div style='font-size:22px; font-weight:bold; color:#ffd166; margin-bottom:8px;'>
            🔎 مدل {st.session_state.model_type} پیش‌بینی کرده است که احتمال تأخیر پرواز:
            <span style='color:#ef476f;'>{pred_prob*100:.2f}%</span> است.
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

       
#------------------------------------------
import streamlit as st
import os

def about_us_page():
    st.title("👥 درباره ما")
    st.write("این پروژه توسط تیمی از متخصصان در حوزه‌های مختلف توسعه داده شده است:")

    # مسیر پوشه عکس‌ها (فرض کنیم پوشه images کنار app.py باشه)
    img_path = "images"

    team_members = [
        {
            "role": "توسعه‌دهنده یادگیری ماشین ",
            "desc": "طراحی و پیاده‌سازی مدل‌های پیش‌بینی پروازها، پردازش داده و بهینه‌سازی الگوریتم‌ها.",
            "icon": "🤖",
            "color": "#e0f7fa",
            "img": os.path.join(img_path, "1.png")
        },
        {
            "role": "توسعه‌دهنده Full Stack",
            "desc": "توسعه سمت سرور، معماری نرم‌افزار و یکپارچه‌سازی رابط کاربری با سرویس‌های داده.",
            "icon": "💻",
            "color": "#f1f8e9",
            "img": os.path.join(img_path, "2.png")
        },
        {
            "role": "مدیر پروژه",
            "desc": "مدیریت تیم، تحلیل نیازهای کسب‌وکار و اطمینان از دستیابی به اهداف پروژه.",
            "icon": "📊",
            "color": "#f1f8e9",
            "img": os.path.join(img_path, "3.png")
        }
    ]

    cols = st.columns(len(team_members))
    for col, member in zip(cols, team_members):
        with col:
            if os.path.exists(member['img']):
               st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #fff7f0, #ffe0b2);
                        padding:20px;
                        border-radius:15px;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
                        text-align:center;
                        height:320px;
                    ">
                        <img src="data:image/jpg;base64,{get_base64(member['img'])}" 
                            style="width:80px;height:80px;border-radius:50%;margin-bottom:10px;">
                        <div style="font-size:26px;">{member['icon']}</div>
                        <h4 style="color:#ff5722;">{member['role']}</h4>
                        <p style="font-size:14px; color:#333; line-height:1.6;">{member['desc']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.error(f"❌ تصویر یافت نشد: {member['img']}")

# تابع برای تبدیل عکس به base64 (برای نمایش در Streamlit HTML)
import base64
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ----------------- Charts -----------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def gregorian_to_jalali(date):
    gy = date.year
    gm = date.month
    gd = date.day
    g_d_m = [0,31,59,90,120,151,181,212,243,273,304,334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    if gm > 2:
        gy2 = gy + 1
    else:
        gy2 = gy
    days = 365*gy + (gy2+3)//4 - (gy2+99)//100 + (gy2+399)//400 - 80 + gd + g_d_m[gm-1]
    jy += 33*(days//12053)
    days %= 12053
    jy += 4*(days//1461)
    days %= 1461
    if days > 365:
        jy += (days-1)//365
        days = (days-1)%365
    if days < 186:
        jm = 1 + days//31
        jd = 1 + days%31
    else:
        jm = 7 + (days-186)//30
        jd = 1 + (days-186)%30
    return f"{jy}/{jm:02d}/{jd:02d}"

def show_dashboard_charts(df_local, start_date=None, end_date=None):
    if df_local.empty:
        st.warning("داده‌ای برای نمایش وجود ندارد!")
        return

    required_columns = ["DelayMinutes", "Delayed", "Airline", "Weekday", "ScheduledHour"]
    missing_cols = [col for col in required_columns if col not in df_local.columns]
    if missing_cols:
        st.error(f"ستون‌های مورد نیاز وجود ندارند: {missing_cols}")
        return

    if "FlightDate" not in df_local.columns:
        df_local["FlightDate"] = pd.date_range(start="2023-01-01", periods=len(df_local), freq="D")
    df_local["FlightDate"] = pd.to_datetime(df_local["FlightDate"])

    # تعیین بازه پیشفرض اگر کاربر مشخص نکرده
    if start_date is None:
        start_date = df_local["FlightDate"].min().date()
    if end_date is None:
        end_date = df_local["FlightDate"].max().date()

    st.markdown(f"**بازه زمانی: {gregorian_to_jalali(start_date)} تا {gregorian_to_jalali(end_date)}**")

    # اعمال فیلتر روی DataFrame
    df_local = df_local[(df_local["FlightDate"].dt.date >= start_date) & 
                        (df_local["FlightDate"].dt.date <= end_date)]

    if df_local.empty:
        st.warning("هیچ داده‌ای در بازه زمانی انتخاب شده وجود ندارد!")
        return

    # محاسبات گج‌ها
    avg_delay = df_local["DelayMinutes"].mean()
    total_flights = len(df_local)
    delayed_ratio = (df_local["Delayed"].sum() / total_flights) * 100

    col1, col2, col3 = st.columns(3)
    gauges = [
        {"value": avg_delay, "title": "میانگین تأخیر", "suffix": " دقیقه", "range": [0, 60],
         "steps": [(0, 15, "#6ee7b7"), (15, 30, "#fde68a"), (30, 60, "#fca5a5")]},
        {"value": delayed_ratio, "title": "درصد تأخیر", "suffix": "%", "range": [0, 100],
         "steps": [(0, 20, "#6ee7b7"), (20, 50, "#fde68a"), (50, 100, "#fca5a5")]},
        {"value": total_flights, "title": "کل پروازها", "suffix": "", "range": [0, total_flights*1.2],
         "steps": [(0, total_flights/3, "#6ee7b7"),
                   (total_flights/3, total_flights*0.66, "#fde68a"),
                   (total_flights*0.66, total_flights*1.2, "#fdba74")]}
    ]

    for col, g in zip((col1, col2, col3), gauges):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(g["value"], 1),
            number={'suffix': g["suffix"], 'font': {'size': 28, 'color': "white"}},
            title={'text': g["title"].replace(" ", "\n"), 'font': {'size': 18, 'color': "white"}, 'align': 'center'},
            gauge={'axis': {'range': g["range"]},
                   'bar': {'color': "#3b82f6", 'thickness': 0.5},
                   'steps': [{'range': [s[0], s[1]], 'color': s[2]} for s in g["steps"]],
                   'bgcolor': "rgba(0,0,0,0)"}
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=20, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
        col.plotly_chart(fig, use_container_width=True)

    # نمودار ایرلاین‌ها
    col4, col5 = st.columns(2)
    airline_delay = df_local.groupby("Airline")["DelayMinutes"].mean().reset_index()
    fig_airline = px.bar(
        airline_delay, x="Airline", y="DelayMinutes",
        title="میانگین تأخیر بر اساس ایرلاین",
        color="DelayMinutes", color_continuous_scale=["#6ee7b7", "#fde68a", "#fca5a5"],
        text_auto=True
    )
    fig_airline.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
    col4.plotly_chart(fig_airline, use_container_width=True)

    # نمودار روزهای هفته
    weekday_count = df_local["Weekday"].value_counts().reindex(
        ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"], fill_value=0
    ).reset_index()
    weekday_count.columns = ["Weekday", "Count"]
    fig_weekday = px.bar(
        weekday_count, x="Weekday", y="Count",
        title="تعداد پروازها بر اساس روز هفته",
        color="Count", color_continuous_scale="Blues",
        text_auto=True
    )
    fig_weekday.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
    col5.plotly_chart(fig_weekday, use_container_width=True)

    st.markdown("---")

    # نمودار ساعت پروازها
    col6, col7 = st.columns(2)
    hour_count = df_local["ScheduledHour"].value_counts().sort_index().reset_index()
    hour_count.columns = ["Hour", "Count"]
    fig_hour = px.bar(hour_count, x="Hour", y="Count", title="توزیع پروازها بر اساس ساعت", color="Count",
                      color_continuous_scale="Viridis", text_auto=True)
    fig_hour.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
    col6.plotly_chart(fig_hour, use_container_width=True)

    # میانگین تأخیر بر اساس ساعت
    delay_by_hour = df_local.groupby("ScheduledHour")["DelayMinutes"].mean().reset_index()
    fig_delay_hour = px.line(delay_by_hour, x="ScheduledHour", y="DelayMinutes",
                             title="میانگین تأخیر بر اساس ساعت پرواز", markers=True)
    fig_delay_hour.update_traces(line_color="#3b82f6", marker=dict(size=8, color="#1f2937"))
    fig_delay_hour.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(dtick=1))
    col7.plotly_chart(fig_delay_hour, use_container_width=True)

# ----------------- Pages Router -----------------
current = st.session_state.current_page
# داده نمونه (یک بار در سطح بالا)
airlines = ["ایران ایر", "ماهان", "قشم", "آسمان"]
np.random.seed(42)
data = []
for airline in airlines:
    flight_count = np.random.randint(5, 61)
    delays = np.random.randint(0, 61, size=flight_count)
    for delay in delays:
        data.append({
            "Airline": airline,
            "DelayMinutes": delay,
            "Delayed": int(delay > 0),
            "Weekday": np.random.choice(["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]),
            "ScheduledHour": np.random.randint(0, 24)
        })

df = pd.DataFrame(data)
# صفحه ورود
if current == "login":
    import base64

    # 📌 پس‌زمینه (عکس هواپیما)
    def set_background_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        img_base64 = base64.b64encode(data).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .login-card {{
            width: 350px;
            background: rgba(255,255,255,0.95);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
            text-align: center;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background_image("4.jpg")

    # 📌 کادر ورود
    with st.container():
        # st.markdown('<div class="login-card">', unsafe_allow_html=True)

        st.markdown("<h3>🔐 ورود به سیستم</h3>", unsafe_allow_html=True)

        username = st.text_input("نام کاربری", key="login_user")
        password = st.text_input("رمز عبور", type="password", key="login_pass")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("ورود"):
                USERS = {
                    "admin": {"password": "123", "role": "admin"},
                    "user": {"password": "123", "role": "user"}
                }
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.role = USERS[username]["role"]
                    goto("معرفی")
                    st.success("✅ ورود موفق بود!")
                    st.rerun()
                else:
                    st.error("❌ نام کاربری یا رمز عبور اشتباه است.")
        with c2:
            if st.button("انصراف"):
                goto("معرفی")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# شروع پیش‌بینی
elif current == "فرم پیش‌بینی":
    prediction_page()

# تاریخچه
elif current == "سوابق پیش‌بینی":
    st.markdown("<h2>📋 سوابق پیش‌بینی‌های انجام شده</h2>", unsafe_allow_html=True)
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history)
    else:
        st.write("هیچ سابقه‌ای موجود نیست.")

elif current == "مقایسه ایرلاین‌ها":
    
    st.markdown("<h2 style='text-align:right;'>📊 مقایسه میانگین تأخیر (دقیقه) و تعداد پروازها</h2>", unsafe_allow_html=True)

        # داده‌های نمونه
    airlines = ["ایران ایر", "ماهان", "قشم", "آسمان"]
    np.random.seed(42)

    data = []
    for airline in airlines:
        flight_count = np.random.randint(5, 61)  # تعداد پرواز بین 5 تا 60
        delays = np.random.randint(0, 61, size=flight_count)  # تأخیر بین 0 تا 60 دقیقه
        for delay in delays:
            data.append({"ایرلاین": airline, "تاخیر_دقیقه": delay})

    df = pd.DataFrame(data)

    # محاسبه میانگین تأخیر و تعداد پروازها
    avg_delay = df.groupby('ایرلاین')['تاخیر_دقیقه'].mean().reset_index(name='میانگین_تاخیر')
    flight_counts = df['ایرلاین'].value_counts().reset_index()
    flight_counts.columns = ['ایرلاین', 'تعداد_پرواز']
    comparison_df = pd.merge(avg_delay, flight_counts, on='ایرلاین')

    # نمودار با گرادیانت نارنجی
    fig = px.bar(
        comparison_df,
        x='ایرلاین',
        y='میانگین_تاخیر',
        color='تعداد_پرواز',
        color_continuous_scale=px.colors.sequential.Oranges,  # گرادیانت نارنجی
        labels={'میانگین_تاخیر': 'میانگین تأخیر (دقیقه)', 'ایرلاین': 'ایرلاین', 'تعداد_پرواز': 'تعداد پروازها'},
        # title="مقایسه میانگین تأخیر (دقیقه)و تعداد پروازها بر اساس ایرلاین"
    )

    # تنظیمات ظاهری
    fig.update_layout(
    # title={
    #     'text': "مقایسه میانگین تأخیر و تعداد پروازها بر اساس ایرلاین",
    #     'x': 0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top',
    #     'font': dict(family="Arial, sans-serif", size=24, color="white")
    # },
    xaxis_title="ایرلاین",
    yaxis_title="میانگین تأخیر (دقیقه)",
    font=dict(family="Arial, sans-serif", size=16, color="black"),
    xaxis=dict(tickfont=dict(size=16, family="Arial, sans-serif", color="black")),
    yaxis=dict(tickfont=dict(size=16, family="Arial, sans-serif", color="black")),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    shapes=[  
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, x1=1, y0=0, y1=1,
            fillcolor="rgba(255, 200, 0, 0.1)",  
            layer="below",
            line_width=0,
        )
    ]
    )


    st.plotly_chart(fig, use_container_width=True) 






elif current == "داشبورد تحلیل پروازها":
    st.markdown("<h2>📋 داشبورد تحلیل پروازها</h2>", unsafe_allow_html=True)
    show_dashboard_charts(df)

elif current == "درباره ما":
    about_us_page()


elif current == "تنظیمات ادمین":
    st.title("⚙️ تنظیمات")

    if st.session_state.get("role") == "admin":
        st.subheader("🔧 بخش مدیریت ادمین")
        
    # بخش ثبت داده‌ها برای همه
    data_source = st.selectbox("انتخاب منبع داده:", ["CSV/Excel", "API"])

    file = None
    api_url = None

    if data_source == "CSV/Excel":
        file = st.file_uploader("📂 فایل CSV یا Excel خود را انتخاب کنید", type=["csv", "xlsx"], key="file_uploader")
    elif data_source == "API":
        api_url = st.text_input("🔗 آدرس API را وارد کنید", key="api_input")

    if st.button("📊 ثبت داده‌ها", key="register_btn"):
        if data_source == "CSV/Excel":
            if file is not None:
                st.session_state.data_source = "CSV/Excel"
                st.session_state.data_info = file.name
                st.success(f"✅ فایل {file.name} با موفقیت ثبت شد.")
            else:
                st.warning("⚠️ لطفاً یک فایل انتخاب کنید.")
        elif data_source == "API":
            if api_url:
                st.session_state.data_source = "API"
                st.session_state.data_info = api_url
                st.success(f"✅ آدرس API ثبت شد: {api_url}")
            else:
                st.warning("⚠️ لطفاً یک آدرس API وارد کنید.")

