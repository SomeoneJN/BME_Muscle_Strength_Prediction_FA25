# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:25:59 2025

@author: nerij
"""

# main_script.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from noise_and_plot_functions import add_noise_simple, create_noisy_shape, compute_means_pair, plot_comparison, plot_means

# ---------------------
# Define Column Groups
# ---------------------
# For Bottom Cylinder: inner_y and inner_z columns
bottom_x_cols     = [f'inner_y{i}' for i in range(1, 10)]
bottom_y_cols     = [f'inner_z{i}' for i in range(1, 10)]
# For Shape Data:
innerShape_x_cols = [f'innerShape_x{i}' for i in range(1, 10)]
innerShape_y_cols = [f'innerShape_y{i}' for i in range(1, 10)]
outerShape_x_cols = [f'outerShape_x{i}' for i in range(1, 10)]
outerShape_y_cols = [f'outerShape_y{i}' for i in range(1, 10)]

# ---------------------
# Load Data and Split
# ---------------------
# Update the file path as needed.
data_file = "2025_3_3_intermediate.csv"
df = pd.read_csv(data_file)

# Split data into train and test (80/20 split).
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Make copies for normal and noisy versions.
train_normal = train_df.copy()
test_normal  = test_df.copy()
train_noisy  = train_df.copy()
test_noisy   = test_df.copy()

# ---------------------
# Compute Noise Levels for Bottom Cylinder
# ---------------------
# For inner_y: use only the first 1000 rows.
subset_inner_y = train_normal[bottom_x_cols].iloc[:1000]
noise_inner_y = round(subset_inner_y.std(axis=0).mean(), 2)
print("Noise level for inner_y:", noise_inner_y)

# For inner_z: use only the first 1000 rows.
subset_inner_z = train_normal[bottom_y_cols].iloc[:1000]
noise_inner_z = round(subset_inner_z.std(axis=0).mean(), 2)
print("Noise level for inner_z:", noise_inner_z)

# ---------------------
# Apply Noise to Bottom Cylinder Columns (Train and Test)
# ---------------------
for col in bottom_x_cols:
    train_noisy[col] = add_noise_simple(train_df[col].to_numpy(), noise_inner_y)
    test_noisy[col]  = add_noise_simple(test_df[col].to_numpy(), noise_inner_y)
for col in bottom_y_cols:
    train_noisy[col] = add_noise_simple(train_df[col].to_numpy(), noise_inner_z)
    test_noisy[col]  = add_noise_simple(test_df[col].to_numpy(), noise_inner_z)

# ---------------------
# Process innerShape and outerShape with Circular Spline Noise (noise level 0.1)
# ---------------------
for i in range(len(train_df)):
    row = train_df.iloc[i]
    # Process innerShape:
    x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(innerShape_x_cols):
        train_noisy.at[train_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(innerShape_y_cols):
        train_noisy.at[train_df.index[i], col] = y_noisy[j]
    # Process outerShape:
    x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(outerShape_x_cols):
        train_noisy.at[train_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(outerShape_y_cols):
        train_noisy.at[train_df.index[i], col] = y_noisy[j]

for i in range(len(test_df)):
    row = test_df.iloc[i]
    # Process innerShape:
    x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(innerShape_x_cols):
        test_noisy.at[test_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(innerShape_y_cols):
        test_noisy.at[test_df.index[i], col] = y_noisy[j]
    # Process outerShape:
    x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level=0.1)
    for j, col in enumerate(outerShape_x_cols):
        test_noisy.at[test_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(outerShape_y_cols):
        test_noisy.at[test_df.index[i], col] = y_noisy[j]

# ---------------------
# Limit Number of Curves for Figures 1 and 2:
# Only plot 10 examples for each of the Train and Test comparisons.
# ---------------------
train_normal_subset = train_normal.sample(n=100, random_state=42)
train_noisy_subset  = train_noisy.sample(n=100, random_state=42)
test_normal_subset  = test_normal.sample(n=100, random_state=42)
test_noisy_subset   = test_noisy.sample(n=100, random_state=42)

# ---------------------
# Figure 1: Normal vs Noisy for Train (10 examples)
# ---------------------
fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
plot_comparison(axs1[0], train_normal_subset, train_noisy_subset, bottom_x_cols, bottom_y_cols, "Train: Bottom Cylinder")
plot_comparison(axs1[1], train_normal_subset, train_noisy_subset, innerShape_x_cols, innerShape_y_cols, "Train: Inner Shape")
plot_comparison(axs1[2], train_normal_subset, train_noisy_subset, outerShape_x_cols, outerShape_y_cols, "Train: Outer Shape")
fig1.suptitle("Figure 1: Normal vs Noisy for Train (10 Examples)", fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------
# Figure 2: Normal vs Noisy for Test (10 examples)
# ---------------------
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
plot_comparison(axs2[0], test_normal_subset, test_noisy_subset, bottom_x_cols, bottom_y_cols, "Test: Bottom Cylinder")
plot_comparison(axs2[1], test_normal_subset, test_noisy_subset, innerShape_x_cols, innerShape_y_cols, "Test: Inner Shape")
plot_comparison(axs2[2], test_normal_subset, test_noisy_subset, outerShape_x_cols, outerShape_y_cols, "Test: Outer Shape")
fig2.suptitle("Figure 2: Normal vs Noisy for Test (10 Examples)", fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------
# Figure 3: 5 Mean Curves (Normal vs Noisy) for Train
# ---------------------
train_bottom_means_x, train_bottom_means_y = compute_means_pair(train_normal, bottom_x_cols, bottom_y_cols, groups=5)
train_inner_means_x,  train_inner_means_y  = compute_means_pair(train_normal, innerShape_x_cols, innerShape_y_cols, groups=5)
train_outer_means_x,  train_outer_means_y  = compute_means_pair(train_normal, outerShape_x_cols, outerShape_y_cols, groups=5)
train_bottom_noisy_means_x, train_bottom_noisy_means_y = compute_means_pair(train_noisy, bottom_x_cols, bottom_y_cols, groups=5)
train_inner_noisy_means_x,  train_inner_noisy_means_y  = compute_means_pair(train_noisy, innerShape_x_cols, innerShape_y_cols, groups=5)
train_outer_noisy_means_x,  train_outer_noisy_means_y  = compute_means_pair(train_noisy, outerShape_x_cols, outerShape_y_cols, groups=5)
train_noisy_examples = train_noisy.sample(n=5, random_state=42)

fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
plot_means(axs3[0], train_bottom_means_x, train_bottom_means_y, train_bottom_noisy_means_x, train_bottom_noisy_means_y, train_noisy_examples, bottom_x_cols, bottom_y_cols, "Train: Bottom Cylinder Means")
plot_means(axs3[1], train_inner_means_x, train_inner_means_y, train_inner_noisy_means_x, train_inner_noisy_means_y, train_noisy_examples, innerShape_x_cols, innerShape_y_cols, "Train: Inner Shape Means")
plot_means(axs3[2], train_outer_means_x, train_outer_means_y, train_outer_noisy_means_x, train_outer_noisy_means_y, train_noisy_examples, outerShape_x_cols, outerShape_y_cols, "Train: Outer Shape Means")
fig3.suptitle("Figure 3: 5 Mean Curves (Normal vs Noisy) for Train", fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------
# Figure 4: 5 Mean Curves (Normal vs Noisy) for Test
# ---------------------
test_bottom_means_x, test_bottom_means_y = compute_means_pair(test_normal, bottom_x_cols, bottom_y_cols, groups=5)
test_inner_means_x,  test_inner_means_y  = compute_means_pair(test_normal, innerShape_x_cols, innerShape_y_cols, groups=5)
test_outer_means_x,  test_outer_means_y  = compute_means_pair(test_normal, outerShape_x_cols, outerShape_y_cols, groups=5)
test_bottom_noisy_means_x, test_bottom_noisy_means_y = compute_means_pair(test_noisy, bottom_x_cols, bottom_y_cols, groups=5)
test_inner_noisy_means_x,  test_inner_noisy_means_y  = compute_means_pair(test_noisy, innerShape_x_cols, innerShape_y_cols, groups=5)
test_outer_noisy_means_x,  test_outer_noisy_means_y  = compute_means_pair(test_noisy, outerShape_x_cols, outerShape_y_cols, groups=5)
test_noisy_examples = test_noisy.sample(n=5, random_state=42)

fig4, axs4 = plt.subplots(1, 3, figsize=(18, 6))
plot_means(axs4[0], test_bottom_means_x, test_bottom_means_y, test_bottom_noisy_means_x, test_bottom_noisy_means_y, test_noisy_examples, bottom_x_cols, bottom_y_cols, "Test: Bottom Cylinder Means")
plot_means(axs4[1], test_inner_means_x, test_inner_means_y, test_inner_noisy_means_x, test_inner_noisy_means_y, test_noisy_examples, innerShape_x_cols, innerShape_y_cols, "Test: Inner Shape Means")
plot_means(axs4[2], test_outer_means_x, test_outer_means_y, test_outer_noisy_means_x, test_outer_noisy_means_y, test_noisy_examples, outerShape_x_cols, outerShape_y_cols, "Test: Outer Shape Means")
fig4.suptitle("Figure 4: 5 Mean Curves (Normal vs Noisy) for Test", fontsize=16)
plt.tight_layout()
plt.show()
