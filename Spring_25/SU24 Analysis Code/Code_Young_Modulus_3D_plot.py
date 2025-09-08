# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 02:40:03 2025

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
    """Function to plot actual vs predicted values."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 18))
    colors = ['blue', 'green', 'orange']
    
    for i in range(3):
        axs[i].scatter(y_test[:, i], y_pred[:, i], color=colors[i])
        axs[i].plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], color='red')
        axs[i].set_xlabel(f'Actual Youngs_Modulus{i+1}')
        axs[i].set_ylabel(f'Predicted Youngs_Modulus{i+1}')
        axs[i].set_title(f'Youngs_Modulus{i+1}: Actual vs Predicted {title_suffix}')
    
    plt.tight_layout()
    plt.show()

def plot_prediction_difference(y_test, y_pred):
    """Function to plot the difference between actual and predicted values."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 18))
    colors = ['blue', 'green', 'orange']
    
    for i in range(3):
        axs[i].scatter(100*(y_test[:, i], (y_pred[:, i] - y_test[:, i]) / y_test[:, i]), color=colors[i])
        axs[i].set_xlabel(f'Actual Youngs_Modulus{i+1}')
        axs[i].set_ylabel(f'Prediction Error for Youngs_Modulus{i+1}')
        axs[i].set_ylim((-5, 5))
        axs[i].set_title(f'Youngs_Modulus{i+1}: Actual vs Predicted Difference')
    
    plt.tight_layout()
    plt.show()

# Load data
# df = pd.read_csv("train_data.csv")  # Replace with your actual file path
df = pd.read_csv("2025_2_24_intermediate_2025_2_25_modified_train.csv")  # Replace with your actual file path

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
print(f'Mean Squared Error: {mse}')

# Plot results
# plot_actual_vs_predicted(y_test, y_pred)
# plot_prediction_difference(y_test, y_pred)

# Select three features to vary
feature_indices = [0, 1, 2]  # Indices for 'inner_y1', 'inner_y2', 'inner_y3'
fixed_values = np.mean(X_train, axis=0)  # Keep other features constant

# Generate a grid of values for the first two features
x_range = np.linspace(min(X_train[:, feature_indices[0]]), max(X_train[:, feature_indices[0]]), 20)
y_range = np.linspace(min(X_train[:, feature_indices[1]]), max(X_train[:, feature_indices[1]]), 20)

X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Select fixed values for the third feature
z_fixed_values = np.linspace(min(X_train[:, feature_indices[2]]), max(X_train[:, feature_indices[2]]), 3)

# Plot results for each Young's modulus
for modulus_idx, title, color in zip(range(3), ['Youngs_Modulus1', 'Youngs_Modulus2', 'Youngs_Modulus3'], ['Blues', 'Greens', 'Oranges']):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    for ax, z_fixed in zip(axes, z_fixed_values):
        # Prepare data for prediction
        X_plot = np.tile(fixed_values, (X_grid.size, 1))  # Copy fixed values for all features
        X_plot[:, feature_indices[0]] = X_grid.ravel()
        X_plot[:, feature_indices[1]] = Y_grid.ravel()
        X_plot[:, feature_indices[2]] = z_fixed  # Fix third feature

        # Transform features using polynomial transformation
        X_plot_poly = poly.transform(X_plot)

        # Predict Young’s modulus values
        y_plot = model.predict(X_plot_poly)[:, modulus_idx]  # Select correct modulus output

        # Reshape to match the 2D grid
        Y_surface = y_plot.reshape(X_grid.shape)

        # Create a 3D surface plot
        surf = ax.plot_surface(X_grid, Y_grid, Y_surface, cmap=color, edgecolor='k', alpha=0.7)

        # Labels
        ax.set_xlabel('inner_y1')
        ax.set_ylabel('inner_y2')
        ax.set_zlabel(title)

        # Add fixed inner_y3 value in subplot title
        ax.set_title(f'Fixed inner_y3 = {z_fixed:.2f}', fontsize=12)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

    # Overall figure title
    fig.suptitle(f'Effect of inner_y1 & inner_y2 on {title}', fontsize=16, fontweight='bold')

    # Adjust layout to avoid overlap
    plt.subplots_adjust(top=0.85, wspace=0.3)

    plt.show()















