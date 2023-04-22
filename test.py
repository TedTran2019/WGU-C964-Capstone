# Doing the majority of work in VSCode before transferring to Jupyter Notebook
# Since VSCode has debugger, error messages, and other useful tools
# %%
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, metrics, model_selection
import numpy as np
import pandas as pd

# Obtain and load the data
# Looking at the original dataset compared to this processed one
# Categorical features removed, missing values were filled, some features combined, 
# some features were scaled, and others removed entirely
housing = fetch_california_housing(as_frame=True).frame

# %%
# Simple single feature linear regression example
def single_feature_linear_regression(housing):
    # Decide features and target
    features = housing[['MedInc']]
    target = housing[['MedHouseVal']]
    # Split the data into training and test sets
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=964)
    lin_model = linear_model.LinearRegression()
    lin_model.fit(features_train, target_train)
    target_prediction = lin_model.predict(features_test)
    lin_RMSE = metrics.mean_squared_error(
        target_test, target_prediction, squared=False)
    print(lin_RMSE * 100000) # Roughly 83k

single_feature_linear_regression(housing)
  
# %%
# Random Forest Regression with a little more complexity
def random_forest_regression(features, target):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=964)
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(features_train, target_train)
    target_prediction = random_forest_model.predict(features_test)
    rf_RMSE = metrics.mean_squared_error(
        target_test, target_prediction, squared=False)
    comparison = pd.DataFrame({'Actual': target_test.values.flatten(), 'Predicted': target_prediction.flatten()})
    print(comparison)
    print(rf_RMSE * 100000) # Roughly 51k all features w/ simplified dataset

features = housing.drop('MedHouseVal', axis=1)
target = housing[['MedHouseVal']]
random_forest_regression(features, target)

# %% 
housing.info() # No missing values and all features are numeric
housing.head() # First 5 rows of the data

# AveOccup, Population, AveRooms, and AveBedrms have insane maximum values
# A histogram would show that the data is skewed
housing.describe() # Describes the dataset

# A heatmap or scatter plots would be nice to show correlations as well
# I have no idea what features are correlated with the target yet
correlation_matrix = housing.corr()
print(correlation_matrix)
print(correlation_matrix['MedHouseVal'].sort_values(ascending=False))

# %%
print(fetch_california_housing(as_frame=True).DESCR)

# Fine tuning a model? Use GridSearchCV
