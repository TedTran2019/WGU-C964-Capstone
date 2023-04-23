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
def random_forest_regression(features, target, random_forest_model=RandomForestRegressor()):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=964)
    random_forest_model.fit(features_train, target_train)
    importances = random_forest_model.feature_importances_
    importances = pd.DataFrame({'feature': features.columns, 'importance': np.round(importances, 3)})
    print(importances.sort_values('importance', ascending=False).set_index('feature'))
    target_prediction = random_forest_model.predict(features_test)
    rf_RMSE = metrics.mean_squared_error(
        target_test, target_prediction, squared=False)
    comparison = pd.DataFrame({'Actual': target_test.values.flatten(), 'Predicted': target_prediction.flatten()})
    print(comparison)
    print(rf_RMSE * 100000) # Roughly 51k

# All features have an importance score of more than .01, so I won't drop any
features = housing.drop('MedHouseVal', axis=1)
target = housing[['MedHouseVal']]
random_forest_regression(features, target)

# %% 
print(fetch_california_housing(as_frame=True).DESCR)
housing.info() # No missing values and all features are numeric
housing.head() # First 5 rows of the data

# AveOccup, Population, AveRooms, and AveBedrms have insane maximum values
# A histogram or distribution plot would show that the data is skewed
housing.describe() # Describes the dataset

# A heatmap or scatter plots would be nice to show correlations as well
# I have no idea what features are correlated with the target yet
correlation_matrix = housing.corr()
print(correlation_matrix)
print(correlation_matrix['MedHouseVal'].sort_values(ascending=False))

# %%
# Fine tuning a model? Use GridSearchCV or RandomizedSearchCV
# Grid ensures optimal, but can take forever. Randomized is faster, but not optimal
def grid_search(features, target):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=964)
    random_forest_model = RandomForestRegressor()
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search = model_selection.GridSearchCV(random_forest_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(features_train, target_train)
    print(grid_search.best_params_)
    print(np.sqrt(-grid_search.best_score_)) # RMSE
    return grid_search.best_estimator_

tuned_random_forest = grid_search(features, target) # max_features: 2 and n_estimators: 30 is optimal
# %%
random_forest_regression(features, target, tuned_random_forest) # Consistently silghtly better, RMSE of 50k

# %%
