# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 00:29:25 2025

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

# Sample 150 sets of the original data from each, if available.
if len(train_df) > 150:
    train_sample = train_df.sample(n=150, random_state=42)
else:
    train_sample = train_df.copy()
if len(test_df) > 150:
    test_sample = test_df.sample(n=150, random_state=42)
else:
    test_sample = test_df.copy()

# ---------------------------
# Step 4: Compute Mean Curves from 5 Equal Groups
# ---------------------------
def compute_group_means(data, columns, groups=5):
    """
    Splits the DataFrame into 'groups' nearly equal parts and computes the mean
    for the specified columns for each part.
    Returns a numpy array of shape (groups, number_of_columns).
    """
    splitted = np.array_split(data, groups)
    means = [grp[columns].mean() for grp in splitted]
    return np.array([mean.values for mean in means])

# Bottom Cylinder means
train_bottom_x = compute_group_means(train_df, bottom_x_cols, groups=5)
train_bottom_y = compute_group_means(train_df, bottom_y_cols, groups=5)
test_bottom_x = compute_group_means(test_df, bottom_x_cols, groups=5)
test_bottom_y = compute_group_means(test_df, bottom_y_cols, groups=5)

# Inner Shape means
train_inner_x = compute_group_means(train_df, inner_shape_x_cols, groups=5)
train_inner_y = compute_group_means(train_df, inner_shape_y_cols, groups=5)
test_inner_x = compute_group_means(test_df, inner_shape_x_cols, groups=5)
test_inner_y = compute_group_means(test_df, inner_shape_y_cols, groups=5)

# Outer Shape means
train_outer_x = compute_group_means(train_df, outer_shape_x_cols, groups=5)
train_outer_y = compute_group_means(train_df, outer_shape_y_cols, groups=5)
test_outer_x = compute_group_means(test_df, outer_shape_x_cols, groups=5)
test_outer_y = compute_group_means(test_df, outer_shape_y_cols, groups=5)

# ---------------------------
# Step 5: Plotting
# ---------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# ----- Plot 1: Bottom Cylinder Points -----
# Plot original curves (150 sets) for Train
first_train_orig = True
for idx, (_, row) in enumerate(train_sample.iterrows()):
    x = row[bottom_x_cols].values.astype(float)
    y = row[bottom_y_cols].values.astype(float)
    if first_train_orig:
        axs[0].plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.5, label='Train Original')
        first_train_orig = False
    else:
        axs[0].plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.5)

# Plot original curves (150 sets) for Test
first_test_orig = True
for idx, (_, row) in enumerate(test_sample.iterrows()):
    x = row[bottom_x_cols].values.astype(float)
    y = row[bottom_y_cols].values.astype(float)
    if first_test_orig:
        axs[0].plot(x, y, marker='s', linestyle='--', color='red', alpha=0.5, label='Test Original')
        first_test_orig = False
    else:
        axs[0].plot(x, y, marker='s', linestyle='--', color='red', alpha=0.5)

# Plot mean curves (5 groups) with thicker, darker lines
first_train_mean = True
for i in range(train_bottom_x.shape[0]):
    x = train_bottom_x[i, :]
    y = train_bottom_y[i, :]
    if first_train_mean:
        axs[0].plot(x, y, marker='o', linestyle='-', color='navy', linewidth=3, label='Train Mean')
        first_train_mean = False
    else:
        axs[0].plot(x, y, marker='o', linestyle='-', color='navy', linewidth=3)
        
first_test_mean = True
for i in range(test_bottom_x.shape[0]):
    x = test_bottom_x[i, :]
    y = test_bottom_y[i, :]
    if first_test_mean:
        axs[0].plot(x, y, marker='s', linestyle='--', color='darkred', linewidth=3, label='Test Mean')
        first_test_mean = False
    else:
        axs[0].plot(x, y, marker='s', linestyle='--', color='darkred', linewidth=3)

axs[0].set_title('Bottom Cylinder Points')
axs[0].set_xlabel('inner_y')
axs[0].set_ylabel('inner_z')
axs[0].legend(loc='upper right')

# ----- Plot 2: Inner Radius Points (innerShape) -----
# Plot original curves (150 sets) for Train (inner shape)
first_train_orig = True
for idx, (_, row) in enumerate(train_sample.iterrows()):
    x = row[inner_shape_x_cols].values.astype(float)
    y = row[inner_shape_y_cols].values.astype(float)
    # Compute a periodic (closed) spline for smooth curve
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_train_orig:
        axs[1].plot(x, y, 'o', color='blue', alpha=0.5, label='Train Original')
        axs[1].plot(spline[0], spline[1], '-', color='blue', alpha=0.5)
        first_train_orig = False
    else:
        axs[1].plot(x, y, 'o', color='blue', alpha=0.5)
        axs[1].plot(spline[0], spline[1], '-', color='blue', alpha=0.5)

# Plot original curves (150 sets) for Test (inner shape)
first_test_orig = True
for idx, (_, row) in enumerate(test_sample.iterrows()):
    x = row[inner_shape_x_cols].values.astype(float)
    y = row[inner_shape_y_cols].values.astype(float)
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_test_orig:
        axs[1].plot(x, y, 's', color='red', alpha=0.5, label='Test Original')
        axs[1].plot(spline[0], spline[1], '--', color='red', alpha=0.5)
        first_test_orig = False
    else:
        axs[1].plot(x, y, 's', color='red', alpha=0.5)
        axs[1].plot(spline[0], spline[1], '--', color='red', alpha=0.5)

# Plot mean curves (5 groups) for inner shape with thicker, darker lines
first_train_mean = True
for i in range(train_inner_x.shape[0]):
    x = train_inner_x[i, :]
    y = train_inner_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_train_mean:
        axs[1].plot(x, y, 'o', color='navy', linewidth=3, label='Train Mean')
        axs[1].plot(spline[0], spline[1], '-', color='navy', linewidth=3)
        first_train_mean = False
    else:
        axs[1].plot(x, y, 'o', color='navy', linewidth=3)
        axs[1].plot(spline[0], spline[1], '-', color='navy', linewidth=3)
        
first_test_mean = True
for i in range(test_inner_x.shape[0]):
    x = test_inner_x[i, :]
    y = test_inner_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_test_mean:
        axs[1].plot(x, y, 's', color='darkred', linewidth=3, label='Test Mean')
        axs[1].plot(spline[0], spline[1], '--', color='darkred', linewidth=3)
        first_test_mean = False
    else:
        axs[1].plot(x, y, 's', color='darkred', linewidth=3)
        axs[1].plot(spline[0], spline[1], '--', color='darkred', linewidth=3)

axs[1].set_title('Inner Radius Points')
axs[1].set_xlabel('innerShape_x')
axs[1].set_ylabel('innerShape_y')
axs[1].legend(loc='upper right')

# ----- Plot 3: Outer Radius Points (outerShape) -----
# Plot original curves (150 sets) for Train (outer shape)
first_train_orig = True
for idx, (_, row) in enumerate(train_sample.iterrows()):
    x = row[outer_shape_x_cols].values.astype(float)
    y = row[outer_shape_y_cols].values.astype(float)
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_train_orig:
        axs[2].plot(x, y, 'o', color='blue', alpha=0.5, label='Train Original')
        axs[2].plot(spline[0], spline[1], '-', color='blue', alpha=0.5)
        first_train_orig = False
    else:
        axs[2].plot(x, y, 'o', color='blue', alpha=0.5)
        axs[2].plot(spline[0], spline[1], '-', color='blue', alpha=0.5)

# Plot original curves (150 sets) for Test (outer shape)
first_test_orig = True
for idx, (_, row) in enumerate(test_sample.iterrows()):
    x = row[outer_shape_x_cols].values.astype(float)
    y = row[outer_shape_y_cols].values.astype(float)
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_test_orig:
        axs[2].plot(x, y, 's', color='red', alpha=0.5, label='Test Original')
        axs[2].plot(spline[0], spline[1], '--', color='red', alpha=0.5)
        first_test_orig = False
    else:
        axs[2].plot(x, y, 's', color='red', alpha=0.5)
        axs[2].plot(spline[0], spline[1], '--', color='red', alpha=0.5)

# Plot mean curves (5 groups) for outer shape with thicker, darker lines
first_train_mean = True
for i in range(train_outer_x.shape[0]):
    x = train_outer_x[i, :]
    y = train_outer_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_train_mean:
        axs[2].plot(x, y, 'o', color='navy', linewidth=3, label='Train Mean')
        axs[2].plot(spline[0], spline[1], '-', color='navy', linewidth=3)
        first_train_mean = False
    else:
        axs[2].plot(x, y, 'o', color='navy', linewidth=3)
        axs[2].plot(spline[0], spline[1], '-', color='navy', linewidth=3)
        
first_test_mean = True
for i in range(test_outer_x.shape[0]):
    x = test_outer_x[i, :]
    y = test_outer_y[i, :]
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    if first_test_mean:
        axs[2].plot(x, y, 's', color='darkred', linewidth=3, label='Test Mean')
        axs[2].plot(spline[0], spline[1], '--', color='darkred', linewidth=3)
        first_test_mean = False
    else:
        axs[2].plot(x, y, 's', color='darkred', linewidth=3)
        axs[2].plot(spline[0], spline[1], '--', color='darkred', linewidth=3)

axs[2].set_title('Outer Radius Points')
axs[2].set_xlabel('outerShape_x')
axs[2].set_ylabel('outerShape_y')
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.show()
