import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_line, labs, theme_minimal
import pgeocode
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
testing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')


# Initialize the zip code lookup for US
nomi = pgeocode.Nominatim("us")

# Get city and state from the zip code
housing[['city', 'state']] = housing['zipcode'].apply(lambda x: pd.Series(nomi.query_postal_code(x)[['place_name', 'state_name']]))

# Compute average price per zipcode
avg_price_per_zip = housing.groupby("zipcode")["price"].mean().reset_index()
avg_price_per_zip.rename(columns={"price": "average_price"}, inplace=True)

# Merge the average price into the dataset
housing = housing.merge(avg_price_per_zip, on="zipcode", how="left", suffixes=("", "_dup"))

X = housing[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms','bedrooms', 'floors', 'waterfront', 'view', 'sqft_basement', 'average_price','yr_renovated', 'yr_built']]
y = housing['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBRegressor(
        n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions


# Compute metrics
mae = mean_absolute_error(y_test, predictions)  # MAE
mse = mean_squared_error(y_test, predictions)   # MSE
rmse = np.sqrt(mse)  # RMSE
rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(predictions)))  # RMSLE
r2 = r2_score(y_test, predictions)  # R² Score

# Print results
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"RMSLE: {rmsle:.4f}")
print(f"R² Score: {r2:.4f}")