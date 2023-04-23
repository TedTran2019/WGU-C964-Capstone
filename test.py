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

correlation_matrix = housing.corr()

# All features have an importance score of more than .01, so I won't drop any
features = housing.drop('MedHouseVal', axis=1)
target = housing['MedHouseVal']

def regression(features, target, model):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=964)
    model.fit(features_train, target_train) # Unnecessary if used grid_search
    # Importances can be plotted with a bar chart
    # importances = model.feature_importances_
    # importances = pd.DataFrame({'feature': features.columns, 'importance': np.round(importances, 3)})
    # print(importances.sort_values('importance', ascending=False).set_index('feature'))
    target_prediction = model.predict(features_test)
    model_RMSE = metrics.mean_squared_error(
        target_test, target_prediction, squared=False)
    # comparison = pd.DataFrame({'Actual': target_test.values.flatten(), 'Predicted': target_prediction.flatten()})
    # print(comparison)
    avg_error = round(model_RMSE * 100000, 2)
    print(f"On average, house estimation values are ${avg_error:.2f} off.")

# %% 
print(fetch_california_housing(as_frame=True).DESCR)
# %%
housing.info() # No missing values and all features are numeric
# %%
housing.head() # First 5 rows of the data
# %%
# AveOccup, Population, AveRooms, and AveBedrms have insane maximum values
# A histogram or distribution plot would show that the data is skewed
housing.describe() # Describes the dataset

# A heatmap or scatter plots would be nice to show correlations as well
# %%
print(correlation_matrix)
# %%
print(correlation_matrix['MedHouseVal'].sort_values(ascending=False))

# %%
# Fine tuning a model? Use GridSearchCV or RandomizedSearchCV
# Grid ensures optimal, but can take forever. Randomized is faster, but not optimal
def grid_search(features, target, model, param_grid):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=964)
    grid_search = model_selection.GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(features_train, target_train)
    print(grid_search.best_params_)
    print(np.sqrt(-grid_search.best_score_)) # RMSE
    return grid_search.best_estimator_

# %%
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
tuned_random_forest = grid_search(features, target, RandomForestRegressor(), param_grid)  # max_features: 2 and n_estimators: 30 is optimal

# %%
regression(housing[['MedInc']], target, RandomForestRegressor()) # RMSE is 97k
regression(features, target, RandomForestRegressor()) # Untuned, 51k
regression(features, target, tuned_random_forest) # Consistently better, 50k

# %% 
regression(housing[['MedInc']], target, linear_model.LinearRegression()) # 83k
regression(features, target, linear_model.LinearRegression()) # 72k
