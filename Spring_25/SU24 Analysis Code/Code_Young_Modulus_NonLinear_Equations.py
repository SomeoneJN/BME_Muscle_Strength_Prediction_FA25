# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 03:17:37 2025

@author: nerij
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def plot_actual_vs_predicted(y_test, y_pred, title_suffix=""):
    """Plot actual vs predicted Young's modulus values."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))
    colors = ['blue', 'green', 'orange']

    for i, color in enumerate(colors):
        min_val, max_val = min(y_test[:, i].min(), y_pred[:, i].min()), max(y_test[:, i].max(), y_pred[:, i].max())
        
        axs[i].scatter(y_test[:, i], y_pred[:, i], color=color, alpha=0.6)
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle="--")
        axs[i].set_xlabel(f'Actual Youngs_Modulus{i+1}')
        axs[i].set_ylabel(f'Predicted Youngs_Modulus{i+1}')
        axs[i].set_title(f'Youngs_Modulus{i+1}: Actual vs Predicted {title_suffix}')
        axs[i].set_xlim(min_val, max_val)
        axs[i].set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.show()

def plot_prediction_difference(y_test, y_pred):
    """Plot the difference between actual and predicted values."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))
    colors = ['blue', 'green', 'orange']

    for i, color in enumerate(colors):
        error = (y_pred[:, i] - y_test[:, i]) / y_test[:, i]
        axs[i].scatter(y_test[:, i], error, color=color, alpha=0.6)
        axs[i].axhline(y=0, color='red', linestyle="--")
        axs[i].set_xlabel(f'Actual Youngs_Modulus{i+1}')
        axs[i].set_ylabel('Prediction Error')
        axs[i].set_ylim((-0.01, 0.01))
        axs[i].set_title(f'Youngs_Modulus{i+1}: Prediction Error')

    plt.tight_layout()
    plt.show()

# Load data
df = pd.read_csv("train_data.csv")  # Replace with actual file path


# Define features and target
# X = df[['inner_y1', 'inner_y2', 'inner_y3', 'outer_y1', 'outer_y2', 'outer_y3', 'innerShape_x1', 'innerShape_x2', 'innerShape_x3']].values
X = df[['Principal Component 1 Inner Radius','Principal Component 2 Inner Radius','Principal Component 3 Inner Radius','Principal Component 1 Outer Radius','Principal Component 2 Outer Radius','Principal Component 3 Outer Radius','Principal Component 1 Bottom Cylinder','Principal Component 2 Bottom Cylinder','Principal Component 3 Bottom Cylinder']].values

y = df[['Part1_E', 'Part3_E', 'Part11_E']].values  # Adjust column names as per dataset

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Evaluate
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print(f'Mean Squared Error: {mse:.6f}')

## Polynomial
# Get the polynomial feature names
feature_names = poly.get_feature_names_out()

# Loop through each Young's modulus output
for i in range(y.shape[1]):
    coefficients = model.coef_[i]  # Coefficients for Young's Modulus i
    intercept = model.intercept_[i]  # Intercept term
    
    # Construct the polynomial equation
    equation_terms = [f"{coeff:.4f} * {feature}" for coeff, feature in zip(coefficients, feature_names)]
    equation = " + ".join(equation_terms)
    
    print(f"\nBest-Fit Polynomial Equation for Young's Modulus {i+1}:")
    print(f"Y = {intercept:.4f} + {equation}")
