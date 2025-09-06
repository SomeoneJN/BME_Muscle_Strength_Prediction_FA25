# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:46:46 2025

@author: nerij
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.model_selection import train_test_split

# --- Step 1: Read CSV data ---
data_file = "2025_3_3_intermediate.csv"
df = pd.read_csv(data_file)

# --- Define the column names by group ---
# Bottom Cylinder points: inner_y* for x and inner_z* for y
bottom_x_cols = [f'inner_y{i}' for i in range(1, 10)]
bottom_y_cols = [f'inner_z{i}' for i in range(1, 10)]

# Inner Radius points (inner shape): innerShape_x* and innerShape_y*
inner_shape_x_cols = [f'innerShape_x{i}' for i in range(1, 10)]
inner_shape_y_cols = [f'innerShape_y{i}' for i in range(1, 10)]

# Outer Radius points (outer shape): outerShape_x* and outerShape_y*
outer_shape_x_cols = [f'outerShape_x{i}' for i in range(1, 10)]
outer_shape_y_cols = [f'outerShape_y{i}' for i in range(1, 10)]

# --- Step 2: Split data into train and test sets ---
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Only select 100 random samples from each set (if available)
if len(train_df) > 100:
    train_df = train_df.sample(n=10, random_state=42)
if len(test_df) > 100:
    test_df = test_df.sample(n=10, random_state=42)

# --- Step 3: Create a figure with 3 subplots for the different point groups ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# -----------------------
# Plot 1: Bottom Cylinder Points (inner_y vs inner_z)
# -----------------------
for i, (_, row) in enumerate(train_df.iterrows()):
    x = row[bottom_x_cols].values.astype(float)
    y = row[bottom_y_cols].values.astype(float)
    if i == 0:
        axs[0].plot(x, y, marker='o', linestyle='-', color='blue', label='Train')
    else:
        axs[0].plot(x, y, marker='o', linestyle='-', color='blue')

for i, (_, row) in enumerate(test_df.iterrows()):
    x = row[bottom_x_cols].values.astype(float)
    y = row[bottom_y_cols].values.astype(float)
    if i == 0:
        axs[0].plot(x, y, marker='s', linestyle='--', color='red', label='Test')
    else:
        axs[0].plot(x, y, marker='s', linestyle='--', color='red')

axs[0].set_title('Bottom Cylinder points')
axs[0].set_xlabel('inner_y values')
axs[0].set_ylabel('inner_z values')
axs[0].legend(loc='upper right')  # Fixed location for legend

# -----------------------
# Plot 2: Inner Radius Points with circular spline
# -----------------------
first_train_inner = True
for _, row in train_df.iterrows():
    x = row[inner_shape_x_cols].values.astype(float)
    y = row[inner_shape_y_cols].values.astype(float)
    # Create a periodic (closed) spline to smooth the circle
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if first_train_inner:
        axs[1].plot(x, y, 'o', color='blue', label='Train Points')
        axs[1].plot(spline[0], spline[1], '-', color='blue', label='Train Spline')
        first_train_inner = False
    else:
        axs[1].plot(x, y, 'o', color='blue')
        axs[1].plot(spline[0], spline[1], '-', color='blue')

first_test_inner = True
for _, row in test_df.iterrows():
    x = row[inner_shape_x_cols].values.astype(float)
    y = row[inner_shape_y_cols].values.astype(float)
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if first_test_inner:
        axs[1].plot(x, y, 's', color='red', label='Test Points')
        axs[1].plot(spline[0], spline[1], '--', color='red', label='Test Spline')
        first_test_inner = False
    else:
        axs[1].plot(x, y, 's', color='red')
        axs[1].plot(spline[0], spline[1], '--', color='red')

axs[1].set_title('Inner Radius points')
axs[1].set_xlabel('X coordinate')
axs[1].set_ylabel('Y coordinate')
axs[1].legend(loc='upper right')  # Fixed location for legend

# -----------------------
# Plot 3: Outer Radius Points with circular spline
# -----------------------
first_train_outer = True
for _, row in train_df.iterrows():
    x = row[outer_shape_x_cols].values.astype(float)
    y = row[outer_shape_y_cols].values.astype(float)
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if first_train_outer:
        axs[2].plot(x, y, 'o', color='blue', label='Train Points')
        axs[2].plot(spline[0], spline[1], '-', color='blue', label='Train Spline')
        first_train_outer = False
    else:
        axs[2].plot(x, y, 'o', color='blue')
        axs[2].plot(spline[0], spline[1], '-', color='blue')

first_test_outer = True
for _, row in test_df.iterrows():
    x = row[outer_shape_x_cols].values.astype(float)
    y = row[outer_shape_y_cols].values.astype(float)
    tck, _ = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, 200)
    spline = splev(unew, tck)
    
    if first_test_outer:
        axs[2].plot(x, y, 's', color='red', label='Test Points')
        axs[2].plot(spline[0], spline[1], '--', color='red', label='Test Spline')
        first_test_outer = False
    else:
        axs[2].plot(x, y, 's', color='red')
        axs[2].plot(spline[0], spline[1], '--', color='red')

axs[2].set_title('Outer Radius points')
axs[2].set_xlabel('X coordinate')
axs[2].set_ylabel('Y coordinate')
axs[2].legend(loc='upper right')  # Fixed location for legend

plt.tight_layout()
plt.show()
