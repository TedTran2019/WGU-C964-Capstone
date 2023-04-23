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
random_state = 964

# All features have an importance score of more than .01, so I won't drop any
features = housing.drop('MedHouseVal', axis=1)
min_features = housing[['MedInc', 'AveRooms', 'Latitude', 'Longitude']]
bare_min_features = housing[['MedInc', 'Latitude', 'Longitude']]
target = housing['MedHouseVal']

def regression(features, target, model):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=random_state)
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

# %%
# A heatmap or scatter plots would be nice to show correlations as well
print(correlation_matrix)
# %%
print(correlation_matrix['MedHouseVal'].sort_values(ascending=False))

# %%
# Fine tuning a model? Use GridSearchCV or RandomizedSearchCV
# Grid ensures optimal, but can take forever. Randomized is faster, but not optimal
def grid_search(features, target, model, param_grid):
    features_train, features_test, target_train, target_test = model_selection.train_test_split(
        features, target, test_size=0.3, random_state=random_state)
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
tuned_random_forest = grid_search(features, target, RandomForestRegressor(random_state=random_state), param_grid)  # max_features: 2 and n_estimators: 30 is optimal
tuned_min_random_forest = grid_search(min_features, target, RandomForestRegressor(random_state=random_state), param_grid) # max_features: 4 and n_estimators: 30 is optimal
tuned_bare_min_random_forest = grid_search(bare_min_features, target, RandomForestRegressor(random_state=random_state), param_grid) # max_features: 4 and n_estimators: 30 is optimal
# %%
regression(housing[['MedInc']], target, RandomForestRegressor(random_state=random_state)) # RMSE is 97k
regression(features, target, RandomForestRegressor(random_state=random_state)) # Untuned, 51k
regression(features, target, tuned_random_forest) # Consistently better tuned, 50k
regression(min_features, target, RandomForestRegressor(random_state=random_state)) # Untuned 48.5k
regression(min_features, target, tuned_min_random_forest) # Tuned 49k
regression(bare_min_features, target, RandomForestRegressor(random_state=random_state)) # 49k
regression(bare_min_features, target, tuned_bare_min_random_forest) # 49.5k

# %% 
regression(housing[['MedInc']], target, linear_model.LinearRegression()) # 83k
regression(features, target, linear_model.LinearRegression()) # 72k
regression(min_features, target, linear_model.LinearRegression()) # 73k
regression(bare_min_features, target, linear_model.LinearRegression()) # 73k

# %%
def get_user_input():
    while True:
        try:
            rooms = float(input("Enter number of rooms: "))
            if rooms < 0:
                raise ValueError("Number of rooms can't be negative.")
            break
        except ValueError as e:
            print(f"Error: {e}")

    while True:
        try:
            lat = float(input("Enter latitude(32.5 - 42): "))
            if lat < 32.5 or lat > 42:
                raise ValueError("Latitude must be between 32.5 and 42.")
            break
        except ValueError as e:
            print(f"Error: {e}")

    while True:
        try:
            lng = float(input("Enter longitude(-124.65 - -114.13): "))
            if lng < -124.65 or lng > -114.13:
                raise ValueError("Longitude must be between -124.65 and -114.13.")
            break
        except ValueError as e:
            print(f"Error: {e}")

    while True:
        try:
            income = float(input("Enter yearly income of current/previous household: "))
            if income < 0:
                raise ValueError("Income can't be negative.")
            break
        except ValueError as e:
            print(f"Error: {e}")

    return rooms, lat, lng, income / 10000

# %% 
def predict_property_value():
    rooms, lat, lng, income = get_user_input()
    user_input = pd.DataFrame({'MedInc': [income], 'AveRooms': [rooms], 'Latitude': [lat], 'Longitude': [lng]})
    prediction = tuned_min_random_forest.predict(user_input)
    print(f"Your house is estimated to be worth ${prediction[0] * 100000:.2f} dollars.")

# %%
predict_property_value()
