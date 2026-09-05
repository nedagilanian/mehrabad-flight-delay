# ✈️ Mehrabad Flight Delay Prediction

A machine learning and deep learning project for predicting flight delays using a synthetic dataset inspired by Mehrabad Airport operations.

## 📌 Project Overview

This project analyzes flight delay patterns and predicts whether a flight will be delayed based on operational factors such as:

* Airline
* Destination
* Day of Week
* Scheduled Departure Hour

The project includes:

* Dataset Generation
* Exploratory Data Analysis (EDA)
* Data Visualization
* GRU-based Deep Learning Model
* Model Evaluation

---

## 📂 Project Structure

```text
mehrabad-flight-delay/
│
├── generate_dataset.py
├── gru_model.py
├── evaluate_model.py
├── eda.py
├── mehrabad_flights.csv
├── gru_delay_model.h5
│
├── delay_by_airline.png
├── delay_by_destination.png
├── delay_by_hour.png
├── delay_distribution.png
│
└── README.md
```

---

## 📊 Exploratory Data Analysis

### Average Delay by Airline

![Delay by Airline](delay_by_airline.png)

### Average Delay by Destination

![Delay by Destination](delay_by_destination.png)

### Average Delay by Scheduled Hour

![Delay by Hour](delay_by_hour.png)

### Delay Distribution

![Delay Distribution](delay_distribution.png)

---

## 🤖 Deep Learning Model

The project uses a GRU (Gated Recurrent Unit) neural network implemented with TensorFlow/Keras.

### Features

* Airline
* Destination
* Weekday
* Scheduled Hour

### Target

* Delayed (0 = No Delay, 1 = Delay)

---

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

---

## 📈 Model Evaluation

The GRU model was trained and evaluated on the generated dataset.

Current performance:

* Accuracy ≈ 66%

This project is intended as an educational demonstration of machine learning workflows including data generation, preprocessing, visualization, model training, and evaluation.

---

## 🚀 Future Improvements

* Compare GRU with Random Forest
* Compare GRU with Logistic Regression
* Hyperparameter Tuning
* Real Flight Data Integration
* Interactive Dashboard using Streamlit

---

## 👩‍💻 Author

Neda Gilanian

Computer Engineering 

Interested in Data Analysis, Machine Learning, Deep Learning, and Backend Development.
