import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

model = XGBRegressor()
model.load_model("model_xgb.json")
scaler = joblib.load("scaler.joblib")

new_data = np.array([[9000, 2000, 10, 150, 1, 150, 1, 4, 0, 0, 0, 0, 0, 1, 1]])  #['Postal code', 'Construction year', 'Number of rooms', 'Living area','kitchen', 'Primary energy consumption', 'Double glazing','State_encoded', 'Heating type_Electric', 'Heating type_Fuel oil','Heating type_Gas', 'Heating type_Pellet', 'Heating type_Solar','Heating type_Wood', 'Type of property_house']
new_data_scaled = scaler.fit_transform(new_data)

predicted_price = model.predict(new_data_scaled)
print("Predicted Price:", predicted_price[0])