# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 01:56:48 2025

@author: nerij
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.model_selection import train_test_split

# =============================================================================
# Utility functions
# =============================================================================
def add_noise_simple(arr, noise_level=0.1):
    """Add noise to a numpy array using a normal distribution."""
    return arr + np.random.normal(0, noise_level, size=arr.shape)

def create_noisy_shape(row, shape_x_cols, shape_y_cols, noise_level=0.1):
    """
    For a given row, fit a periodic spline to the 9 shape points,
    sample 100 equally spaced points along the spline,
    add noise (σ = noise_level) to these points,
    and then select 9 equidistant points from the 100.
    Returns the 9 noisy x and y values.
    """
    # Convert original columns to numpy arrays (using to_numpy() to avoid .values warnings)
    x = row[shape_x_cols].to_numpy().astype(float)
    y = row[shape_y_cols].to_numpy().astype(float)
    # Fit a periodic (circular) spline
    tck, _ = splprep([x, y], s=0, per=True)
    # Sample 100 equally spaced parameter values
    u_new = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_new, tck)
    # Add noise to the sampled spline points
    x_noisy = x_spline + np.random.normal(0, noise_level, size=x_spline.shape)
    y_noisy = y_spline + np.random.normal(0, noise_level, size=y_spline.shape)
    # Choose 9 equidistant indices from 100 points
    indices = np.linspace(0, 99, 9, dtype=int)
    return x_noisy[indices], y_noisy[indices]

def compute_means_pair(data, cols_x, cols_y, groups=5):
    """
    Splits the DataFrame into nearly equal parts and computes the mean of the specified 
    x and y columns for each group.
    Returns two numpy arrays of shape (groups, len(cols)).
    """
    splitted = np.array_split(data, groups)
    mean_x, mean_y = [], []
    for group in splitted:
        mean_x.append(group[cols_x].mean().to_numpy())
        mean_y.append(group[cols_y].mean().to_numpy())
    return np.array(mean_x), np.array(mean_y)

def get_comparison_subplot(ax, normal_df, noisy_df, cols_x, cols_y, title):
    """
    Plot curves from normal and noisy data onto the given axis.
    Uses data from the specified x and y columns.
    """
    # Plot normal data curves in light blue
    for idx, row in normal_df.iterrows():
        x = row[cols_x].to_numpy().astype(float)
        y = row[cols_y].to_numpy().astype(float)
        ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3)
    # Overlay noisy data curves in light red dashed style
    for idx, row in noisy_df.iterrows():
        x = row[cols_x].to_numpy().astype(float)
        y = row[cols_y].to_numpy().astype(float)
        ax.plot(x, y, marker='s', linestyle='--', color='red', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(['Normal', 'Noisy'], loc='upper right')

def get_means_subplot(ax, normal_means_x, normal_means_y,
                      noisy_means_x, noisy_means_y,
                      noisy_examples, cols_x, cols_y, title):
    """
    Plot mean curves and overlay a few noisy example curves on the given axis.
    """
    # Plot normal mean curves with thick dark blue lines
    for i in range(normal_means_x.shape[0]):
        ax.plot(normal_means_x[i], normal_means_y[i],
                 marker='o', linestyle='-', color='navy', linewidth=3)
    # Plot noisy mean curves with thick dark red dashed lines
    for i in range(noisy_means_x.shape[0]):
        ax.plot(noisy_means_x[i], noisy_means_y[i],
                 marker='s', linestyle='--', color='darkred', linewidth=3)
    # Overlay 5 randomly selected noisy examples using different style
    for idx, row in noisy_examples.iterrows():
        x = row[cols_x].to_numpy().astype(float)
        y = row[cols_y].to_numpy().astype(float)
        ax.plot(x, y, marker='x', linestyle=':', color='orange', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(['Normal Mean', 'Noisy Mean', 'Noisy Example'], loc='upper right')

# =============================================================================
# Main processing: Create Normal and Noisy Data
# =============================================================================
# Define column groups
bottom_x_cols     = [f'inner_y{i}' for i in range(1, 10)]
bottom_y_cols     = [f'inner_z{i}' for i in range(1, 10)]
innerShape_x_cols = [f'innerShape_x{i}' for i in range(1, 10)]
innerShape_y_cols = [f'innerShape_y{i}' for i in range(1, 10)]
outerShape_x_cols = [f'outerShape_x{i}' for i in range(1, 10)]
outerShape_y_cols = [f'outerShape_y{i}' for i in range(1, 10)]

# Load CSV (adjust filepath as needed)
data_file = "2025_3_3_intermediate.csv"
df = pd.read_csv(data_file)

# Split into train and test sets (e.g., 80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Make copies: one for "normal" (original) and one for "noisy"
train_normal = train_df.copy()
test_normal  = test_df.copy()
train_noisy  = train_df.copy()
test_noisy   = test_df.copy()

# ----- For Bottom Cylinder (inner_y and inner_z) -----
# Add noise directly (using the simple noise function)
for col in bottom_x_cols + bottom_y_cols:
    train_noisy[col] = add_noise_simple(train_df[col].to_numpy(), noise_level=0.1)
    test_noisy[col]  = add_noise_simple(test_df[col].to_numpy(), noise_level=0.1)

# ----- For innerShape and outerShape -----
# For each row, create the noisy version based on the circular spline procedure.
for i in range(len(train_df)):
    row = train_df.iloc[i]
    # Process innerShape
    x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(innerShape_x_cols):
        train_noisy.at[train_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(innerShape_y_cols):
        train_noisy.at[train_df.index[i], col] = y_noisy[j]
    # Process outerShape
    x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(outerShape_x_cols):
        train_noisy.at[train_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(outerShape_y_cols):
        train_noisy.at[train_df.index[i], col] = y_noisy[j]

for i in range(len(test_df)):
    row = test_df.iloc[i]
    # Process innerShape
    x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(innerShape_x_cols):
        test_noisy.at[test_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(innerShape_y_cols):
        test_noisy.at[test_df.index[i], col] = y_noisy[j]
    # Process outerShape
    x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(outerShape_x_cols):
        test_noisy.at[test_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(outerShape_y_cols):
        test_noisy.at[test_df.index[i], col] = y_noisy[j]

# =============================================================================
# Figure 1: Compare Normal vs Noisy Data (for Train and Test)
# =============================================================================
fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows (train, test) x 3 columns (bottom, inner, outer)

# Row 0: Train
get_comparison_subplot(axs1[0, 0], train_normal, train_noisy, bottom_x_cols, bottom_y_cols,
                        "Train: Bottom Cylinder (inner_y vs inner_z)")
get_comparison_subplot(axs1[0, 1], train_normal, train_noisy, innerShape_x_cols, innerShape_y_cols,
                        "Train: Inner Shape")
get_comparison_subplot(axs1[0, 2], train_normal, train_noisy, outerShape_x_cols, outerShape_y_cols,
                        "Train: Outer Shape")
# Row 1: Test
get_comparison_subplot(axs1[1, 0], test_normal, test_noisy, bottom_x_cols, bottom_y_cols,
                        "Test: Bottom Cylinder (inner_y vs inner_z)")
get_comparison_subplot(axs1[1, 1], test_normal, test_noisy, innerShape_x_cols, innerShape_y_cols,
                        "Test: Inner Shape")
get_comparison_subplot(axs1[1, 2], test_normal, test_noisy, outerShape_x_cols, outerShape_y_cols,
                        "Test: Outer Shape")

fig1.suptitle("Normal Data vs Noisy Data", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# =============================================================================
# Figure 2: Compare 5 Mean Curves vs 5 Noisy Example Curves (for Train and Test)
# =============================================================================
# Compute group means (5 equal groups) for normal and noisy data.
train_bottom_mean_x, train_bottom_mean_y = compute_means_pair(train_normal, bottom_x_cols, bottom_y_cols, groups=5)
test_bottom_mean_x,  test_bottom_mean_y  = compute_means_pair(test_normal, bottom_x_cols, bottom_y_cols, groups=5)
train_inner_mean_x,  train_inner_mean_y  = compute_means_pair(train_normal, innerShape_x_cols, innerShape_y_cols, groups=5)
test_inner_mean_x,   test_inner_mean_y   = compute_means_pair(test_normal, innerShape_x_cols, innerShape_y_cols, groups=5)
train_outer_mean_x,  train_outer_mean_y  = compute_means_pair(train_normal, outerShape_x_cols, outerShape_y_cols, groups=5)
test_outer_mean_x,   test_outer_mean_y   = compute_means_pair(test_normal, outerShape_x_cols, outerShape_y_cols, groups=5)

train_bottom_noisy_mean_x, train_bottom_noisy_mean_y = compute_means_pair(train_noisy, bottom_x_cols, bottom_y_cols, groups=5)
test_bottom_noisy_mean_x,  test_bottom_noisy_mean_y  = compute_means_pair(test_noisy, bottom_x_cols, bottom_y_cols, groups=5)
train_inner_noisy_mean_x,  train_inner_noisy_mean_y  = compute_means_pair(train_noisy, innerShape_x_cols, innerShape_y_cols, groups=5)
test_inner_noisy_mean_x,   test_inner_noisy_mean_y   = compute_means_pair(test_noisy, innerShape_x_cols, innerShape_y_cols, groups=5)
train_outer_noisy_mean_x,  train_outer_noisy_mean_y  = compute_means_pair(train_noisy, outerShape_x_cols, outerShape_y_cols, groups=5)
test_outer_noisy_mean_x,   test_outer_noisy_mean_y   = compute_means_pair(test_noisy, outerShape_x_cols, outerShape_y_cols, groups=5)

# Pick 5 example curves from the noisy data (randomly sampled)
train_noisy_examples = train_noisy.sample(n=5, random_state=42)
test_noisy_examples  = test_noisy.sample(n=5, random_state=42)

fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows (train, test) x 3 columns (bottom, inner, outer)

# Row 0: Train
get_means_subplot(axs2[0, 0], train_bottom_mean_x, train_bottom_mean_y,
                  train_bottom_noisy_mean_x, train_bottom_noisy_mean_y,
                  train_noisy_examples, bottom_x_cols, bottom_y_cols,
                  "Train: Bottom Cylinder Means vs Noisy Examples")
get_means_subplot(axs2[0, 1], train_inner_mean_x, train_inner_mean_y,
                  train_inner_noisy_mean_x, train_inner_noisy_mean_y,
                  train_noisy_examples, innerShape_x_cols, innerShape_y_cols,
                  "Train: Inner Shape Means vs Noisy Examples")
get_means_subplot(axs2[0, 2], train_outer_mean_x, train_outer_mean_y,
                  train_outer_noisy_mean_x, train_outer_noisy_mean_y,
                  train_noisy_examples, outerShape_x_cols, outerShape_y_cols,
                  "Train: Outer Shape Means vs Noisy Examples")
# Row 1: Test
get_means_subplot(axs2[1, 0], test_bottom_mean_x, test_bottom_mean_y,
                  test_bottom_noisy_mean_x, test_bottom_noisy_mean_y,
                  test_noisy_examples, bottom_x_cols, bottom_y_cols,
                  "Test: Bottom Cylinder Means vs Noisy Examples")
get_means_subplot(axs2[1, 1], test_inner_mean_x, test_inner_mean_y,
                  test_inner_noisy_mean_x, test_inner_noisy_mean_y,
                  test_noisy_examples, innerShape_x_cols, innerShape_y_cols,
                  "Test: Inner Shape Means vs Noisy Examples")
get_means_subplot(axs2[1, 2], test_outer_mean_x, test_outer_mean_y,
                  test_outer_noisy_mean_x, test_outer_noisy_mean_y,
                  test_noisy_examples, outerShape_x_cols, outerShape_y_cols,
                  "Test: Outer Shape Means vs Noisy Examples")

fig2.suptitle("Mean Curves vs Noisy Examples", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
