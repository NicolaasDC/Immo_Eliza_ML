import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib


# Read the csv file and create a dataframe
df = pd.read_csv("./data/clean_data_with_region.csv")

y = df[['price']]
X = df.drop(columns=['price'])

# Delete all columns with more than 30% missing values
for column in X:
    if X[column].isnull().sum(axis = 0) > len(X) * 0.3:
        X = X.drop(columns=[column])
        
# Delete non relevant columns (Energy class uses different values in Flanders and Wallonia)        
X = X.drop(columns=['Property ID', 'Locality name', 'Energy class', 'region', 'Heating type'])

# Impute missing data
imputer_most_frequent = SimpleImputer(strategy='most_frequent')
columns_to_impute_most_frequent = ['Number of rooms', 'kitchen', 'State of builing', 'Double glazing']
imputer_mean = SimpleImputer(strategy='mean')
columns_to_impute_mean = ['Construction year', 'Living area', 'Primary energy consumption']

X[columns_to_impute_most_frequent] = imputer_most_frequent.fit_transform(X[columns_to_impute_most_frequent])
X[columns_to_impute_mean] = imputer_mean.fit_transform(X[columns_to_impute_mean])

# Transform the State of building column to ordinal data
categorie = [['To restore', 'To renovate', 'To be done up', 'Good', 'Just renovated', 'As new']]

encoder = OrdinalEncoder(categories=categorie)
X['State_encoded'] = encoder.fit_transform(X[['State of builing']])

X = X.drop(columns=['State of builing'])

# Create encoder object
enc = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")

# Apply fit method to the data frame

encoded_data = enc.fit_transform(X[['Type of property']])

X = pd.concat([X.drop(columns=['Type of property']).reset_index(drop=True), encoded_data.reset_index(drop=True)], axis=1)

# Convert Number of rooms columns to int values
X['Number of rooms'] = X['Number of rooms'].astype('int')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

# Scale the X sets
scaler = StandardScaler()

joblib.dump(scaler, "scaler.joblib")

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the XGBRegressor model
xgb_model = XGBRegressor(
    objective='reg:squarederror',  # Use square error for regression
    n_estimators=100,              # Number of boosting rounds
    learning_rate=0.2,             # Learning rate
    max_depth=4,                   # Maximum depth of a tree                        
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

xgb_model.save_model("model_xgb.json")

# Evaluate the model
train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)
print("XGBoost Training Score:", train_score)
print("XGBoost Test Score:", test_score)

# Make predictions using the trained model
y_pred_xgb = xgb_model.predict(X_test)

# Calculate Mean Absolute Error
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print("Mean Absolute Error on Test Set (MAE):", mae_xgb)

