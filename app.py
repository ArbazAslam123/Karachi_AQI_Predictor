import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scalers
final_model = joblib.load("final_model.pkl")
X_scaler = joblib.load("x_scaler.pkl")  # Ensure the file path is correct
y_scaler = joblib.load("y_scaler.pkl")

# Define the feature columns (ensure these match the model input features)
feature_cols = ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres', 'coco', 'Day', 'Month', 'Weekday', 'Year']

# Streamlit UI to input features for prediction
st.title("PM10 and PM2.5 Prediction for the Next 3 Days")
st.write("Enter the weather features below:")

# Input fields for user to enter feature values
temp = st.number_input("Temperature (°C)", value=25.0)
dwpt = st.number_input("Dew Point (°C)", value=18.0)
rhum = st.number_input("Humidity (%)", value=60)
wdir = st.number_input("Wind Direction (°)", value=180)
wspd = st.number_input("Wind Speed (m/s)", value=5.0)
pres = st.number_input("Pressure (hPa)", value=1013)
coco = st.selectbox("Cloud Cover (1-10)", options=[i for i in range(1, 11)], index=6)
day = st.number_input("Day of the Month", value=15)
month = st.number_input("Month", value=4)
weekday = st.number_input("Weekday (0=Monday, 6=Sunday)", value=1)
year = st.number_input("Year", value=2025)

# Create a DataFrame from the input data
user_input = pd.DataFrame({
    'temp': [temp],
    'dwpt': [dwpt],
    'rhum': [rhum],
    'wdir': [wdir],
    'wspd': [wspd],
    'pres': [pres],
    'coco': [coco],
    'Day': [day],
    'Month': [month],
    'Weekday': [weekday],
    'Year': [year]
})

# Scale the features using the loaded scaler
scaled_features = X_scaler.transform(user_input)

# Predict the PM10 and PM2.5 for the next 3 days using the final model
future_preds_scaled = final_model.predict(scaled_features)

# Inverse scale the predictions to get them back to the original scale
future_preds = y_scaler.inverse_transform(future_preds_scaled)

# Create a DataFrame for easy viewing
pred_columns = ['PM10_day1', 'PM10_day2', 'PM10_day3', 'PM2.5_day1', 'PM2.5_day2', 'PM2.5_day3']
future_df = pd.DataFrame(future_preds, columns=pred_columns)

# Display the predictions in the app
st.write("### Predicted PM10 and PM2.5 for the Next 3 Days:")
st.write(future_df.T) 
