# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:43:23 2025

@author: nerij
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.model_selection import train_test_split

# ---------------------
# Helper Functions
# ---------------------
def add_noise_simple(arr, noise_level):
    """Add random noise (Gaussian) to an array using the specified noise_level as sigma."""
    return arr + np.random.normal(0, noise_level, size=arr.shape)

def create_noisy_shape(row, shape_x_cols, shape_y_cols, noise_level=0.1):
    """
    Fit a circular (periodic) spline through 9 shape points, sample 100 points along the spline,
    add noise to the sampled points, and then select 9 equidistant points as the noisy shape.
    """
    x = row[shape_x_cols].to_numpy(dtype=float)
    y = row[shape_y_cols].to_numpy(dtype=float)
    tck, _ = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_new, tck)
    x_noisy = x_spline + np.random.normal(0, noise_level, size=x_spline.shape)
    y_noisy = y_spline + np.random.normal(0, noise_level, size=y_spline.shape)
    indices = np.linspace(0, 99, 9, dtype=int)
    return x_noisy[indices], y_noisy[indices]

def compute_means_pair(data, cols_x, cols_y, groups=5):
    """Split the DataFrame into equal parts (groups) and compute column-wise means for the specified columns."""
    splits = np.array_split(data, groups)
    mean_x = [grp[cols_x].mean().to_numpy() for grp in splits]
    mean_y = [grp[cols_y].mean().to_numpy() for grp in splits]
    return np.array(mean_x), np.array(mean_y)

def plot_comparison(ax, normal_df, noisy_df, cols_x, cols_y, title):
    """Plot curves from normal (blue) and noisy (red) data on the axis."""
    for _, row in normal_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3)
    for _, row in noisy_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='s', linestyle='--', color='red', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(['Normal', 'Noisy'], loc='upper right')

def plot_means(ax, norm_mean_x, norm_mean_y, noisy_mean_x, noisy_mean_y, noisy_examples, cols_x, cols_y, title):
    """Plot the 5 mean curves (normal and noisy) along with several noisy example curves."""
    for i in range(norm_mean_x.shape[0]):
        ax.plot(norm_mean_x[i], norm_mean_y[i], marker='o', linestyle='-', color='navy', linewidth=3)
    for i in range(noisy_mean_x.shape[0]):
        ax.plot(noisy_mean_x[i], noisy_mean_y[i], marker='s', linestyle='--', color='darkred', linewidth=3)
    for _, row in noisy_examples.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='x', linestyle=':', color='orange', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(['Normal Mean', 'Noisy Mean', 'Noisy Example'], loc='upper right')

# ---------------------
# Main Processing
# ---------------------
# Define column groups.
bottom_x_cols     = [f'inner_y{i}' for i in range(1, 10)]
bottom_y_cols     = [f'inner_z{i}' for i in range(1, 10)]
innerShape_x_cols = [f'innerShape_x{i}' for i in range(1, 10)]
innerShape_y_cols = [f'innerShape_y{i}' for i in range(1, 10)]
outerShape_x_cols = [f'outerShape_x{i}' for i in range(1, 10)]
outerShape_y_cols = [f'outerShape_y{i}' for i in range(1, 10)]

# Load CSV file (update the file path as needed)
data_file = "2025_3_3_intermediate.csv"
df = pd.read_csv(data_file)

# Split into train and test sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_normal = train_df.copy()
test_normal  = test_df.copy()
train_noisy  = train_df.copy()
test_noisy   = test_df.copy()

# --- Bottom Cylinder Noise ---
# For inner_y (bottom_x_cols): Use only the first 1000 rows.
subset_inner_y = train_normal[bottom_x_cols].iloc[:1000]
noise_inner_y = round(subset_inner_y.std(axis=0).mean(), 2)
print("Noise level for inner_y:", noise_inner_y/10)

# For inner_z (bottom_y_cols): Use only the first 1000 rows.
subset_inner_z = train_normal[bottom_y_cols].iloc[:1000]
noise_inner_z = round(subset_inner_z.std(axis=0).mean(), 2)
print("Noise level for inner_z:", noise_inner_z)

# Apply noise to bottom cylinder columns for train and test.
for col in bottom_x_cols:
    train_noisy[col] = add_noise_simple(train_df[col].to_numpy(), noise_inner_y)
    test_noisy[col]  = add_noise_simple(test_df[col].to_numpy(), noise_inner_y)
for col in bottom_y_cols:
    train_noisy[col] = add_noise_simple(train_df[col].to_numpy(), noise_inner_z)
    test_noisy[col]  = add_noise_simple(test_df[col].to_numpy(), noise_inner_z)

# --- innerShape and outerShape Noise ---
# For each row, apply the circular spline + noise procedure with noise level 0.1.
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
# Plotting Figures
# ---------------------

# Figure 1: Normal vs. Noisy for Train
fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
plot_comparison(axs1[0], train_normal, train_noisy, bottom_x_cols, bottom_y_cols, "Train: Bottom Cylinder")
plot_comparison(axs1[1], train_normal, train_noisy, innerShape_x_cols, innerShape_y_cols, "Train: Inner Shape")
plot_comparison(axs1[2], train_normal, train_noisy, outerShape_x_cols, outerShape_y_cols, "Train: Outer Shape")
fig1.suptitle("Figure 1: Normal vs. Noisy for Train", fontsize=16)
plt.tight_layout()
plt.show()

# Figure 2: Normal vs. Noisy for Test
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
plot_comparison(axs2[0], test_normal, test_noisy, bottom_x_cols, bottom_y_cols, "Test: Bottom Cylinder")
plot_comparison(axs2[1], test_normal, test_noisy, innerShape_x_cols, innerShape_y_cols, "Test: Inner Shape")
plot_comparison(axs2[2], test_normal, test_noisy, outerShape_x_cols, outerShape_y_cols, "Test: Outer Shape")
fig2.suptitle("Figure 2: Normal vs. Noisy for Test", fontsize=16)
plt.tight_layout()
plt.show()

# Figure 3: 5 Mean Curves (Normal vs. Noisy) for Train
tn_bottom_mx, tn_bottom_my = compute_means_pair(train_normal, bottom_x_cols, bottom_y_cols, groups=5)
tn_inner_mx,  tn_inner_my  = compute_means_pair(train_normal, innerShape_x_cols, innerShape_y_cols, groups=5)
tn_outer_mx,  tn_outer_my  = compute_means_pair(train_normal, outerShape_x_cols, outerShape_y_cols, groups=5)
tn_bottom_noisy_mx, tn_bottom_noisy_my = compute_means_pair(train_noisy, bottom_x_cols, bottom_y_cols, groups=5)
tn_inner_noisy_mx,  tn_inner_noisy_my  = compute_means_pair(train_noisy, innerShape_x_cols, innerShape_y_cols, groups=5)
tn_outer_noisy_mx,  tn_outer_noisy_my  = compute_means_pair(train_noisy, outerShape_x_cols, outerShape_y_cols, groups=5)
tn_noisy_examples = train_noisy.sample(n=5, random_state=42)

fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
plot_means(axs3[0], tn_bottom_mx, tn_bottom_my, tn_bottom_noisy_mx, tn_bottom_noisy_my, tn_noisy_examples,
           bottom_x_cols, bottom_y_cols, "Train: Bottom Cylinder Means")
plot_means(axs3[1], tn_inner_mx, tn_inner_my, tn_inner_noisy_mx, tn_inner_noisy_my, tn_noisy_examples,
           innerShape_x_cols, innerShape_y_cols, "Train: Inner Shape Means")
plot_means(axs3[2], tn_outer_mx, tn_outer_my, tn_outer_noisy_mx, tn_outer_noisy_my, tn_noisy_examples,
           outerShape_x_cols, outerShape_y_cols, "Train: Outer Shape Means")
fig3.suptitle("Figure 3: 5 Mean Curves (Normal vs. Noisy) for Train", fontsize=16)
plt.tight_layout()
plt.show()

# Figure 4: 5 Mean Curves (Normal vs. Noisy) for Test
tn_bottom_mx, tn_bottom_my = compute_means_pair(test_normal, bottom_x_cols, bottom_y_cols, groups=5)
tn_inner_mx,  tn_inner_my  = compute_means_pair(test_normal, innerShape_x_cols, innerShape_y_cols, groups=5)
tn_outer_mx,  tn_outer_my  = compute_means_pair(test_normal, outerShape_x_cols, outerShape_y_cols, groups=5)
tn_bottom_noisy_mx, tn_bottom_noisy_my = compute_means_pair(test_noisy, bottom_x_cols, bottom_y_cols, groups=5)
tn_inner_noisy_mx,  tn_inner_noisy_my  = compute_means_pair(test_noisy, innerShape_x_cols, innerShape_y_cols, groups=5)
tn_outer_noisy_mx,  tn_outer_noisy_my  = compute_means_pair(test_noisy, outerShape_x_cols, outerShape_y_cols, groups=5)
tn_noisy_examples = test_noisy.sample(n=5, random_state=42)

fig4, axs4 = plt.subplots(1, 3, figsize=(18, 6))
plot_means(axs4[0], tn_bottom_mx, tn_bottom_my, tn_bottom_noisy_mx, tn_bottom_noisy_my, tn_noisy_examples,
           bottom_x_cols, bottom_y_cols, "Test: Bottom Cylinder Means")
plot_means(axs4[1], tn_inner_mx, tn_inner_my, tn_inner_noisy_mx, tn_inner_noisy_my, tn_noisy_examples,
           innerShape_x_cols, innerShape_y_cols, "Test: Inner Shape Means")
plot_means(axs4[2], tn_outer_mx, tn_outer_my, tn_outer_noisy_mx, tn_outer_noisy_my, tn_noisy_examples,
           outerShape_x_cols, outerShape_y_cols, "Test: Outer Shape Means")
fig4.suptitle("Figure 4: 5 Mean Curves (Normal vs. Noisy) for Test", fontsize=16)
plt.tight_layout()
plt.show()
