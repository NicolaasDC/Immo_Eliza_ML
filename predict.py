import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# Create a new XGBRegressor
model = XGBRegressor()

# load a saved XGBRegressor model
model.load_model("model\model_xgb.json")
# load the saved scaler, to apply the same scaler
scaler = joblib.load("model\scaler.joblib")

# create a new house and scale it
new_house = np.array([9900, 2000, 5, 150, 1, 150, 1, 4,  1]).reshape(1, -1)  #['Postal code', 'Construction year', 'Number of rooms', 'Living area','kitchen', 'Primary energy consumption', 'Double glazing','State_encoded', 'Type of property_house']
new_data_scaled = scaler.fit_transform(new_house)

# predict the price of the new house
predicted_price = model.predict(new_data_scaled)
print("Predicted Price:", predicted_price[0])