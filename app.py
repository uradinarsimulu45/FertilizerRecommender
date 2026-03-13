import streamlit as st
import pickle
import numpy as np

model, le_soil, le_crop, le_fert = pickle.load(
    open("fertilizer_model.pkl", "rb")
)


st.title("🌱 Smart Fertilizer Recommendation System")

st.write("Enter Soil and Crop Parameters")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
moisture = st.number_input("Moisture")
nitrogen = st.number_input("Nitrogen")
potassium = st.number_input("Potassium")
phosphorous = st.number_input("Phosphorous")
rainfall = st.number_input("Rainfall")
ph = st.number_input("pH")

soil = st.selectbox("Soil", le_soil.classes_)
crop = st.selectbox("Crop", le_crop.classes_)

soil_encoded = le_soil.transform([soil])[0]
crop_encoded = le_crop.transform([crop])[0]

if st.button("Predict Fertilizer"):

    input_data = np.array([[temperature, humidity, moisture,
                            soil_encoded, crop_encoded,
                            nitrogen, potassium, phosphorous, rainfall, ph]])

    prediction = model.predict(input_data)

    fertilizer = le_fert.inverse_transform(prediction)

    st.success("Recommended Fertilizer: " + fertilizer[0])