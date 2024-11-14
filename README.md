# Immo_Eliza_ML

# Description
```
The real estate company Immo Eliza asked you to create a machine learning model to predict prices of real estate properties in Belgium.

This project is a continuation of immo-eliza-scraping (https://github.com/NicolaasDC/immo-eliza-scraping) and immo-eliza-data-analysis project (https://github.com/VB1395/immoeliza_data_analysis)

After first scraping 10000 houses from www.immoweb.be and then analyzing the data. The project continues with creating a machine leaning model for the data.
```

# Objectives
```
- Be able to preprocess data for machine learning.
- Be able to apply a linear regression in a real-life context.
- Be able to explore machine learning models for regression.
- Be able to evaluate the performance of a model
```
# Repo structure
```
.
├── data/
│   └── clean_data_with_region.csv
├── .gitignore
├── model/
│   ├── model_xgb.json
│   └── scaler.joblib
├── notebook/
│   └── pipeline.ipynb
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

# Installation

Clone the repository to your local machine. Set up your virtual enviroment and install the packages from the requirements.txt file

https://github.com/NicolaasDC/Immo_Eliza_ML

# Usage

The pipeline.ipynb notebook was to experiment and getting familiar with the code. 

I tested 3 different models in this notebook: 
- Linear regression 
- RandomForestRegressor
- XGBRegressor.

```
LinearRegression Training Score: 0.239
LinearRegression Test Score (R2 score): 0.238
Mean Absolute Error on Test Set (MAE): 148246 euro

Random Forest Training Score: 0.896
Random Forest Test Score (R2 score): 0.657
Mean Absolute Error on Test Set (MAE): 92252 euro

XGBoost Training Score: 0.848
XGBoost Test Score (R2 score): 0.735
Mean Absolute Error on Test Set (MAE): 87485 euro

XGBoost gave the best prediction results for this dataset.

```
```
In the train.py file I wrote a script to preprocess the data and create a XGBoost model. The model and associated scaler are saved in the model folder.

The predict.py file loads the saved model and can be used to predict the price of a new house. 
```
# Timeline

This project took five days for completion. The 5 days included the first 3 days

# Addition info

This project was done as part of the AI Bootcamp at BeCode.org.

