import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('calories.csv')
df = df.drop(['User_ID', 'Body_Temp'], axis=1)
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
print(df)

X = df.iloc[:, :-1] #all but last
y = df.iloc[:, -1]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize for models that need it SVR & KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR Model
def run_svr(X_train_scaled, X_test_scaled, y_train, y_test):
    model = SVR()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print("\nSVR Results:")
    print("R2 Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

# Random Forest
def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\nRandom Forest Results:")
    print("R2 Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

# KNN Regressor
def run_knn(X_train_scaled, X_test_scaled, y_train, y_test):
    model = KNeighborsRegressor()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print("\nKNN Results:")
    print("R2 Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

# Main: run all models
if __name__ == "__main__":
    #need to add user input here
    run_svr(X_train_scaled, X_test_scaled, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)
    run_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    
