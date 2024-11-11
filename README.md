# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: # Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generating synthetic dataset (Univariate)
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise

# Visualizing the dataset
plt.scatter(X, y)
plt.title("Scatter Plot of Dataset")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.show()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Fitting the model to the training data
model.fit(X_train, y_train)

# Getting the model's parameters (slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_

print(f"Model parameters: Slope = {slope[0]:.2f}, Intercept = {intercept[0]:.2f}")

# Making predictions using the trained model
y_pred = model.predict(X_test)

# Evaluating the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) score: {r2:.2f}")

# Visualizing the regression line
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.title("Linear Regression Fit")
plt.xlabel("Feature X")
plt.ylabel("Predicted y")
plt.legend()
plt.show()

RegisterNumber:  212221045003
*/s.leelathesh
```

## Output:
Model parameters: Slope = 3.05, Intercept = 4.07
Mean Squared Error (MSE): 0.96
R-squared (R²) score: 0.96



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
