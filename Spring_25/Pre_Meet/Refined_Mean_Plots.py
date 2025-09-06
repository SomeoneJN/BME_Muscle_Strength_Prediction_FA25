# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 00:07:26 2025

@author: nerij
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.model_selection import train_test_split

# ---------------------------
# Step 1: Load CSV Data
# ---------------------------
data_file = "2025_3_3_intermediate.csv"
df = pd.read_csv(data_file)

# ---------------------------
# Step 2: Define Column Groups
# ---------------------------
# Bottom Cylinder points: use inner_y* (x) and inner_z* (y)
bottom_x_cols = [f'inner_y{i}' for i in range(1, 10)]
bottom_y_cols = [f'inner_z{i}' for i in range(1, 10)]

# Inner Radius points (inner shape)
inner_shape_x_cols = [f'innerShape_x{i}' for i in range(1, 10)]
inner_shape_y_cols = [f'innerShape_y{i}' for i in range(1, 10)]

# Outer Radius points (outer shape)
outer_shape_x_cols = [f'outerShape_x{i}' for i in range(1, 10)]
outer_shape_y_cols = [f'outerShape_y{i}' for i in range(1, 10)]

# ---------------------------
# Step 3: Split Data into Train and Test Sets
# ---------------------------
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# ---------------------------
# Step 4: Define a Function to Compute Means in 10 Groups
# ---------------------------
def compute_group_means(data, columns, groups=10):
    """
    Splits the DataFrame 'data' into 'groups' equal parts,
    and computes the mean for the specified 'columns' in each part.
    Returns a numpy array of shape (groups, number_of_columns).
    """
    splitted = np.array_split(data, groups)
    means = [grp[columns].mean() for grp in splitted]
    # Convert each Series to a numpy array, so final shape becomes (groups, len(columns))
    return np.array([mean.values for mean in means])

# Compute group means for the three sets of points for train and test

# Bottom Cylinder means (each row here will have 9 values)
train_bottom_x = compute_group_means(train_df, bottom_x_cols, groups=10)
train_bottom_y = compute_group_means(train_df, bottom_y_cols, groups=10)
test_bottom_x  = compute_group_means(test_df, bottom_x_cols, groups=10)
test_bottom_y  = compute_group_means(test_df, bottom_y_cols, groups=10)

# Inner Shape means
train_inner_x = compute_group_means(train_df, inner_shape_x_cols, groups=10)
train_inner_y = compute_group_means(train_df, inner_shape_y_cols, groups=10)
test_inner_x  = compute_group_means(test_df, inner_shape_x_cols, groups=10)
test_inner_y  = compute_group_means(test_df, inner_shape_y_cols, groups=10)

# Outer Shape means
train_outer_x = compute_group_means(train_df, outer_shape_x_cols, groups=10)
train_outer_y = compute_group_means(train_df, outer_shape_y_cols, groups=10)
test_outer_x  = compute_group_means(test_df, outer_shape_x_cols, groups=10)
test_outer_y  = compute_group_means(test_df, outer_shape_y_cols, groups=10)

# ---------------------------
# Step 5: Plotting
# ---------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# ---------- Plot 1: Bottom Cylinder Points (inner_y vs inner_z) -----------
# For each group (of 10) plot the 9 mean points
for i in range(train_bottom_x.shape[0]):  # 10 groups
    # Plot train: show line connecting the 9 points
    if i == 0:
        axs[0].plot(train_bottom_x[i, :], train_bottom_y[i, :],
                    marker='o', linestyle='-', color='blue', label='Train')
    else:
        axs[0].plot(train_bottom_x[i, :], train_bottom_y[i, :],
                    marker='o', linestyle='-', color='blue')
        
for i in range(test_bottom_x.shape[0]):
    if i == 0:
        axs[0].plot(test_bottom_x[i, :], test_bottom_y[i, :],
                    marker='s', linestyle='--', color='red', label='Test')
    else:
        axs[0].plot(test_bottom_x[i, :], test_bottom_y[i, :],
                    marker='s', linestyle='--', color='red')

axs[0].set_title('Bottom Cylinder Mean Points')
axs[0].set_xlabel('inner_y (mean values)')
axs[0].set_ylabel('inner_z (mean values)')
axs[0].legend(loc='upper right')

# ---------- Plot 2: Inner Radius Points with Circular Spline -----------
# For each group, compute the spline from the 9 mean points and plot raw points and spline curve.
for i in range(train_inner_x.shape[0]):
    x = train_inner_x[i, :]
    y = train_inner_y[i, :]
    # Compute a periodic (closed) spline
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if i == 0:
        axs[1].plot(x, y, 'o', color='blue', label='Train Mean Points')
        axs[1].plot(spline[0], spline[1], '-', color='blue', label='Train Spline')
    else:
        axs[1].plot(x, y, 'o', color='blue')
        axs[1].plot(spline[0], spline[1], '-', color='blue')

for i in range(test_inner_x.shape[0]):
    x = test_inner_x[i, :]
    y = test_inner_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if i == 0:
        axs[1].plot(x, y, 's', color='red', label='Test Mean Points')
        axs[1].plot(spline[0], spline[1], '--', color='red', label='Test Spline')
    else:
        axs[1].plot(x, y, 's', color='red')
        axs[1].plot(spline[0], spline[1], '--', color='red')

axs[1].set_title('Inner Radius Mean Points')
axs[1].set_xlabel('innerShape_x (mean values)')
axs[1].set_ylabel('innerShape_y (mean values)')
axs[1].legend(loc='upper right')

# ---------- Plot 3: Outer Radius Points with Circular Spline -----------
for i in range(train_outer_x.shape[0]):
    x = train_outer_x[i, :]
    y = train_outer_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if i == 0:
        axs[2].plot(x, y, 'o', color='blue', label='Train Mean Points')
        axs[2].plot(spline[0], spline[1], '-', color='blue', label='Train Spline')
    else:
        axs[2].plot(x, y, 'o', color='blue')
        axs[2].plot(spline[0], spline[1], '-', color='blue')

for i in range(test_outer_x.shape[0]):
    x = test_outer_x[i, :]
    y = test_outer_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if i == 0:
        axs[2].plot(x, y, 's', color='red', label='Test Mean Points')
        axs[2].plot(spline[0], spline[1], '--', color='red', label='Test Spline')
    else:
        axs[2].plot(x, y, 's', color='red')
        axs[2].plot(spline[0], spline[1], '--', color='red')

axs[2].set_title('Outer Radius Mean Points')
axs[2].set_xlabel('outerShape_x (mean values)')
axs[2].set_ylabel('outerShape_y (mean values)')
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.show()
