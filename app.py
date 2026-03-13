import streamlit as st
import pickle
import pandas as pd

# Load model
model, le_soil, le_crop, le_fert = pickle.load(
    open("fertilizer_model.pkl","rb")
)

# Page config
st.set_page_config(page_title="Fertilizer Recommendation", layout="centered")

# Background style
st.markdown(
"""
<style>

.stApp {
    background-color: #000000;
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:white;
}

label {
    color:white !important;
}

div.stButton > button {
    background-color:#4CAF50;
    color:white;
    font-size:18px;
    border-radius:10px;
    padding:10px 25px;
}

div.stButton > button:hover{
    background-color:#2E7D32;
    color:white;
}

</style>
""",
unsafe_allow_html=True
)

# Title
st.markdown('<p class="title">🌱 Smart Fertilizer Recommendation</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Enter Soil and Crop Details</p>', unsafe_allow_html=True)

# Small agriculture image
st.image(
    "crops.jpeg",
width=700
)

st.write("---")

# Input fields
temperature = st.slider("Temperature", 0, 50, 25)
moisture = st.slider("Moisture", 0.0, 1.0, 0.5)
rainfall = st.slider("Rainfall", 0, 300, 150)
ph = st.slider("PH Value", 4.0, 9.0, 6.5)

nitrogen = st.slider("Nitrogen", 0, 150, 50)
phosphorous = st.slider("Phosphorous", 0, 150, 50)
potassium = st.slider("Potassium", 0, 200, 60)
carbon = st.slider("Carbon", 0.0, 5.0, 1.0)

soil = st.selectbox("Soil Type", le_soil.classes_)
crop = st.selectbox("Crop Type", le_crop.classes_)

st.write("")

# Prediction button
if st.button("🌾 Recommend Fertilizer"):

    soil_encoded = le_soil.transform([soil])[0]
    crop_encoded = le_crop.transform([crop])[0]

    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Moisture": moisture,
        "Rainfall": rainfall,
        "PH": ph,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorous,
        "Potassium": potassium,
        "Carbon": carbon,
        "Soil": soil_encoded,
        "Crop": crop_encoded
    }])

    prediction = model.predict(input_data)

    fertilizer = le_fert.inverse_transform(prediction)

    st.success("🌱 Recommended Fertilizer: " + fertilizer[0])