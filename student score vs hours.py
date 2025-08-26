

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("score_updated.csv")
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

print("\nChecking for null values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# Scatter plot - Hours vs Scores
plt.figure(figsize=(6,4))
plt.scatter(df['Hours'], df['Scores'], color='blue', edgecolor='k')
plt.title("Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Correlation
corr = df.corr()
print("\nCorrelation Matrix:\n", corr)

X = df[['Hours']]   # Independent variable
y = df['Scores']    # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept (b0):", model.intercept_)
print("Coefficient (b1):", model.coef_[0])

plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label="Actual Data", edgecolor='k')
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.title("Linear Regression Fit")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


df['Predicted_Scores'] = model.predict(df[['Hours']])

print("\nActual vs Predicted Scores:\n")
print(df)

# Plotting Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(df['Hours'], df['Scores'], color='blue', label="Actual Scores", edgecolor='k')
plt.plot(df['Hours'], df['Predicted_Scores'], color='red', label="Predicted Scores (Line)")
plt.title("Actual vs Predicted Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
