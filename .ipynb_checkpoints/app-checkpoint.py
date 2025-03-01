import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model("heart_disease_model.h5")

# Load scaler (if used during training)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Heart Disease Prediction")
st.write("Enter patient details to predict the probability of heart disease.")

# User Inputs
age = st.slider("Age", 20, 80, 40)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.slider("Number of Major Vessels (ca)", 0, 4, 1)
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prepare input for prediction
input_data = np.array(
    [
        [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]
    ]
)
input_data_scaled = scaler.transform(input_data)  # Apply same scaling as training

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.write(
        f"Prediction: {np.argmax(prediction)} (Higher values indicate higher risk of heart disease)"
    )
