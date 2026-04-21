import streamlit as st
import numpy as np
import joblib

# =========================================
# 📥 Load Saved Files
# =========================================

model = joblib.load("smart_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Smart Manufacturing IoT", layout="centered")

st.title("🏭 Smart Manufacturing Prediction")
st.write("Predict whether maintenance is required based on sensor inputs")

# =========================================
# 🧾 INPUT FIELDS (EDIT BASED ON YOUR DATASET)
# =========================================

# ⚠️ Replace these with your actual feature names
temperature = st.number_input("Temperature")
pressure = st.number_input("Pressure")
humidity = st.number_input("Humidity")
vibration = st.number_input("Vibration")

# Example categorical input (if exists)
machine_type = st.selectbox("Machine Type", ["TypeA", "TypeB", "TypeC"])

# =========================================
# 🔄 Encode categorical input
# =========================================

if "machine_type" in encoders:
    machine_type_encoded = encoders["machine_type"].transform([machine_type])[0]
else:
    machine_type_encoded = 0

# =========================================
# 📊 Prepare Input
# =========================================

input_data = np.array([[temperature, pressure, humidity, vibration, machine_type_encoded]])

# Scale input
input_scaled = scaler.transform(input_data)

# =========================================
# 🔮 Prediction
# =========================================

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Maintenance Required!")
    else:
        st.success("✅ Machine is Operating Normally")
