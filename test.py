# %%
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, metrics, model_selection
import numpy as np
import pandas as pd

# Obtain and load the data, as_frame=True returns a pandas dataframe
housing = fetch_california_housing(as_frame=True).frame

# Decide features and target
features = housing[['MedInc']]
target = housing[['MedHouseVal']]

# Split the data into training and test sets
features_train, features_test, target_train, target_test = model_selection.train_test_split(features, target, test_size=0.3, random_state=964)

lin_model = linear_model.LinearRegression()
lin_model.fit(features_train, target_train)
target_prediction = lin_model.predict(features_test)
log_RMSE = metrics.mean_squared_error(target_test, target_prediction, squared=False)

random_forest_model = RandomForestRegressor()
random_forest_model.fit(features_train, target_train)
target_prediction = random_forest_model.predict(features_test)
rf_RMSE = metrics.mean_squared_error(target_test, target_prediction, squared=False)

# All displays
# housing.info()
# print(features)
# print(target)
print(log_RMSE)
print(rf_RMSE)
