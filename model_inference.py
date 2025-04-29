import mlflow
import mlflow.pyfunc
import pandas as pd

# Load the model from MLflow model registry
model_name = "AQI_XGBoost_Model"  # Replace this with your actual model name
model_version = 1  # Replace this with your actual model version if necessary

# Load the model from the model registry
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# Example input data (ensure this matches the model's input format)
# This is just sample data; replace it with the actual input for your model
input_data = pd.DataFrame({
    'Temperature': [30],  # Replace with your actual features
    'Humidity': [50],     # Replace with your actual features
    # Add other features here as required by your model
})

# Perform prediction using the loaded model
prediction = model.predict(input_data)

# Print or save the prediction
print("Predicted AQI:", prediction)

