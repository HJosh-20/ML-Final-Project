import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

df = pd.read_csv('calories.csv')
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
print(df)
df_test = pd.read_csv('calories_user_data.csv')

'''
testing the importance of duration feature`
df_no_duration = df.drop(columns='Duration')
X = df_no_duration.iloc[:, :-1]
y = df_no_duration.iloc[:, -1]
'''

X = df.iloc[:, :-1] #all but last
y = df.iloc[:, -1]
print(X.shape)
print(y.shape)

#need fig here bc it represents the whole figure (the full canvas),
#and ax is like a specific spot (or subplot) on that canvas
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print(df.corr()['Calories'].sort_values(ascending=False))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize for models that need it SVR & KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR Model
def run_svr(X_train_scaled, X_test_scaled, y_train, y_test, user_df_scaled):
    model = SVR()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print("\nSVR Results:")
    print("R2 Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    pred_svr = model.predict(user_df_scaled)
    print("\nPredicted calories burned (SVR):", round(pred_svr[0], 2))
    plottingML(preds, title="SVR", target_var=y_test, algorithm_model=model, features_names=X.columns, test_features=X_test)

# Random Forest
def run_random_forest(X_train, X_test, y_train, y_test, user_df):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\nRandom Forest Results:")
    print("R2 Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    pred_rf = model.predict(user_df)
    print("Predicted calories burned (Random Forest):", round(pred_rf[0], 2))
    plottingML(preds, title="RF", target_var=y_test, algorithm_model=model, features_names=X.columns, test_features=X_test)

# KNN Regressor
def run_knn(X_train_scaled, X_test_scaled, y_train, y_test, user_df_scaled):
    model = KNeighborsRegressor()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print("\nKNN Results:")
    print("R2 Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    pred_knn = model.predict(user_df_scaled)
    print("Predicted calories burned (KNN):", round(pred_knn[0], 2))
    plottingML(preds, title="KNN", target_var=y_test, algorithm_model=model, features_names=X.columns, test_features=X_test)

# Polynomial Regression
def run_poly(X_train_scaled, X_test_scaled, y_train, y_test, user_df_scaled):
    for i in range(1, 4):
        poly = PolynomialFeatures(degree = i)
        x_train_poly = poly.fit_transform(X_train_scaled)
        x_test_poly = poly.transform(X_test_scaled)
        user_df_poly = poly.transform(user_df_scaled)
        model = LinearRegression().fit(x_train_poly, y_train)
        preds = model.predict(x_test_poly)
        print(f"\nPolynomial Regression (Degree {i}) Results:")
        print("R2 Score:", r2_score(y_test, preds))
        print("MSE:", mean_squared_error(y_test, preds))
        print("MAE:", mean_absolute_error(y_test, preds))
        pred_poly = model.predict(user_df_poly)
        print("Predicted calories burned (Random Forest):", round(pred_poly[0], 2))
   
def get_height_cm():
    while True:
        height_input = input("Enter your height (e.g., 5 foot 2 inches): ").lower()

        match = re.search(r"(\d+)\s*(foot|feet|ft)\s*(\d+)?\s*(inch|in)?", height_input)
        
        if match:
            feet = int(match.group(1))
            inches = int(match.group(3)) if match.group(3) else 0

            total_inches = feet * 12 + inches
            height_cm = total_inches * 2.54
            return round(height_cm,2)
        else:
            print("Invalid format. Please enter height like '5 foot 2 inches'.")

def get_weight_kg():
    while True:
        weight_input = float(input("Enter your weight in pounds (e.g., 180.33): "))

        weight_kg = weight_input * 0.4535924
        return round(weight_kg, 2)

model_residuals = []
model_names = []
def plottingML(preds, title, target_var=None, algorithm_model=None, features_names=None, test_features=None):
    if target_var is not None:
        # Plot actual vs predicted values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=target_var, y=preds, hue=test_features['Gender'].map({0: 'Male', 1: 'Female'}), palette={'Male': 'blue', 'Female': 'gold'})
        plt.plot([min(target_var), max(target_var)], [min(target_var), max(target_var)], linestyle='--', color='red')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted Values for {title}")
        plt.show()

        #residual plots
        residuals = target_var - preds
        #for box plot residual
        model_residuals.append(residuals)
        model_names.append(title)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=preds, y=residuals, hue=test_features['Gender'].map({0: 'Male', 1: 'Female'}), palette={'Male': 'blue', 'Female': 'gold'})
        #at 0 = prefect prediction, above 0 = model is unpredicted (predicted too low)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlim(0, 300)
        plt.xticks(np.arange(0, 301, 50))
        plt.ylim(-50, 150)
        plt.yticks(np.arange(-50, 151, 25))
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title(f"Residual plot for {title}")
        plt.grid(True)
        plt.show()
                 
   
    #imporant features for RF model.
    if title == 'RF':
        important_features = algorithm_model.feature_importances_
        print(f'Important Features Values: {important_features}')
        #we get numeric values the important features then order then in decending order so its in order of most important to least important
        indices = np.argsort(important_features)[::-1]
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(important_features)), important_features[indices])
        plt.xticks(range(len(important_features)), [features_names[i] for i in indices])
        plt.title(f"Feature Importance - {title}")
        plt.tight_layout()
        plt.show()

# Main: run all models
if __name__ == "__main__":
    #test data inputs height=6 foot 3inches(~190cm), weight=207.2345(~94kg)
    gender = input("Please enter your gender: ")
    while True:
        if   gender.strip().lower() == "male":
            gender = 0
            break
        elif gender.strip().lower() == "female":
            gender = 1
            break
        else:
            gender = input("Invalid gender: Please re-enter your gender: ")
    age = int(input("Please enter your age: "))
    while True:
        if age < 20 or age > 79:
            age = int(input("Invalid age: Please re-enter your age: "))
        else: 
            break
    height_cm = get_height_cm()
    while True:
        if height_cm < 123.0 or height_cm > 222.0:
            print("Invalid height")
            height_cm = get_height_cm()
        else: 
            break
    print(f"Your height in centimeters is: {height_cm:.2f} cm")
    weight_kg = get_weight_kg()
    while True:
        if weight_kg < 36.0 or weight_kg > 132.0:
            print("Invalid weight")
            weight_kg = get_weight_kg()
        else: 
            break
    print(f"Your weight in kilograms is: {weight_kg:.2f} kg")
    duration = float(input("Please enter the duration of your workout: "))
    while True:
        if duration < 1 or duration > 30:
            duration = float(input("Invalid duration: Please re-enter your duration of your workout: "))
        else: 
            break
    heart_rate = float(input("please enter your average heart rate during your workout: "))
    while True:
        if heart_rate < 67 or heart_rate > 128:
            heart_rate = float(input("Invalid average heart rate: Please re-enter your average heart rate during your workout: "))
        else: 
            break

    feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate']
    user_data = np.array([gender, age, height_cm, weight_kg, duration, heart_rate]).reshape(1,-1)
    user_df = pd.DataFrame(user_data, columns=feature_names)
    print("User data (Dataframe): ")
    print(user_df)

    user_df_scaled = scaler.transform(user_data)
    print("User data(Dataframe scaled): ", user_df_scaled)

    run_svr(X_train_scaled, X_test_scaled, y_train, y_test, user_df_scaled)
    run_random_forest(X_train, X_test, y_train, y_test, user_df)
    run_knn(X_train_scaled, X_test_scaled, y_train, y_test, user_df_scaled)
    run_poly(X_train_scaled, X_test_scaled, y_train, y_test, user_df_scaled)

    '''
    #residual boxplots all models
    plt.figure(figsize=(10, 6))
    plt.boxplot(model_residuals, labels=model_names)
    plt.title("Comparison of Residuals Across Models")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''
