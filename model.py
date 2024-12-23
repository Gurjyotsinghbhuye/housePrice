import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import kagglehub

# Download latest version
# path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

# print("Path to dataset files:", path)

# Load the datasets
data = pd.read_csv(r'C:\Users\karan\.cache\kagglehub\datasets\yasserh\housing-prices-dataset\versions\1\Housing.csv')
house  = pd.DataFrame(data)
house.info()
print(house)
'''
 #   Column            Non-Null Count  Dtype 
---  ------            --------------  ----- 
 0   price             545 non-null    int64 
 1   area              545 non-null    int64 
 2   bedrooms          545 non-null    int64
 3   bathrooms         545 non-null    int64
 4   stories           545 non-null    int64
 5   mainroad          545 non-null    object
 6   guestroom         545 non-null    object
 7   basement          545 non-null    object
 8   hotwaterheating   545 non-null    object
 9   airconditioning   545 non-null    object
 10  parking           545 non-null    int64
 11  prefarea          545 non-null    object
 12  furnishingstatus  545 non-null    object
'''

# One-hot encode categorical columns
house_encoded = pd.get_dummies(house, columns=[
    'mainroad', 'guestroom', 'basement', 'prefarea',
    'hotwaterheating', 'airconditioning', 'furnishingstatus'
], drop_first=True)

# Remove unrealistic values
house_encoded['Price_per_sqft'] = house_encoded['price'] / house_encoded['area']
house_encoded = house_encoded[
    (house_encoded['Price_per_sqft'] >= 100) & 
    (house_encoded['Price_per_sqft'] <= 100000) & 
    (house_encoded['bedrooms'] <= house_encoded['area'] // 100) & 
    (house_encoded['bedrooms'] >= house_encoded['bathrooms'])
]
house_encoded.drop(columns=['Price_per_sqft'], inplace=True)

# Add a price category column
conditions = [
    (house_encoded['bedrooms'] <= 2) & (house_encoded['bathrooms'] <= 2),
    (house_encoded['bedrooms'].between(3, 4)) & (house_encoded['bathrooms'].between(2, 3)),
    (house_encoded['bedrooms'] > 4) & (house_encoded['bathrooms'] > 3)
]
choices = ['Low Range', 'Mid Range', 'Luxury']
house_encoded['Price_Category'] = np.select(conditions, choices, default='Uncategorized')

# Features and target
X = house_encoded.drop(columns=['price', 'Price_Category'], errors='ignore')
y = house_encoded['price']

# Scale numerical features (optional)
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Expected features:", X.columns.tolist())

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')