# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 00:27:53 2025

@author: nerij
"""

# main_script.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Ver5_noise_and_plot_functions import (
    add_noise_simple, create_noisy_shape, get_smooth_curve, get_circular_spline_from_points,
    get_perfect_circle_line, plot_perfect_circle, compute_means_pair, plot_comparison, plot_means
)

# ---------------------
# Option: Number of Sets to Test (default 16) and Custom Noise Level for Shape Data.
# ---------------------
num_sets = 16  # Modify as desired.
custom_noise_level = 0.3  # Default noise level for noisy shape data.
nrows = math.ceil(math.sqrt(num_sets))
ncols = math.ceil(num_sets / nrows)

# ---------------------
# Define Column Groups
# ---------------------
# For Bottom Cylinder, use inner_z as x-axis and inner_y as y-axis.
bottom_x_cols = [f'inner_z{i}' for i in range(1, 10)]
bottom_y_cols = [f'inner_y{i}' for i in range(1, 10)]
# Inner Shape.
innerShape_x_cols = [f'innerShape_x{i}' for i in range(1, 10)]
innerShape_y_cols = [f'innerShape_y{i}' for i in range(1, 10)]
# Outer Shape.
outerShape_x_cols = [f'outerShape_x{i}' for i in range(1, 10)]
outerShape_y_cols = [f'outerShape_y{i}' for i in range(1, 10)]

# ---------------------
# Load Data
# ---------------------
data_file = "2025_3_3_intermediate.csv"  # Update as needed.
df = pd.read_csv(data_file)

# ---------------------
# Compute Noise Levels for Bottom Cylinder (first 1000 rows).
subset_inner_z = df[bottom_x_cols].iloc[:1000]  # inner_z for x-axis.
noise_inner_z = round(subset_inner_z.std(axis=0).mean(), 2)
print("Noise level for inner_z (x-axis):", noise_inner_z)
subset_inner_y = df[bottom_y_cols].iloc[:1000]  # inner_y for y-axis.
noise_inner_y = round(subset_inner_y.std(axis=0).mean(), 2)
print("Noise level for inner_y (y-axis):", noise_inner_y)

# ---------------------
# Randomly sample num_sets rows from the Total Set.
subset_df = df.sample(n=num_sets, random_state=42).reset_index(drop=True)

# Create copies for each group.
# Bottom Cylinder.
bottom_normal = subset_df.copy()
bottom_noisy = subset_df.copy()
# Inner Shape.
inner_shape_normal = subset_df.copy()
inner_shape_noisy = subset_df.copy()
# Outer Shape.
outer_shape_normal = subset_df.copy()
outer_shape_noisy = subset_df.copy()

# ---------------------
# Apply noise to Bottom Cylinder columns.
for col in bottom_x_cols:
    bottom_noisy[col] = add_noise_simple(subset_df[col].to_numpy(), noise_inner_z)
for col in bottom_y_cols:
    bottom_noisy[col] = add_noise_simple(subset_df[col].to_numpy(), noise_inner_y)

# ---------------------
# Process Inner Shape:
# Normal inner shape: use noise_level=0.
# Noisy inner shape: use noise_level=custom_noise_level.
for i in range(len(subset_df)):
    row = subset_df.iloc[i]
    x_norm, y_norm = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level=0)
    for j, col in enumerate(innerShape_x_cols):
        inner_shape_normal.at[subset_df.index[i], col] = x_norm[j]
    for j, col in enumerate(innerShape_y_cols):
        inner_shape_normal.at[subset_df.index[i], col] = y_norm[j]
    x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level=custom_noise_level)
    for j, col in enumerate(innerShape_x_cols):
        inner_shape_noisy.at[subset_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(innerShape_y_cols):
        inner_shape_noisy.at[subset_df.index[i], col] = y_noisy[j]

# ---------------------
# Process Outer Shape:
# Normal outer shape: use noise_level=0.
# Noisy outer shape: use noise_level=custom_noise_level.
for i in range(len(subset_df)):
    row = subset_df.iloc[i]
    x_norm, y_norm = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level=0)
    for j, col in enumerate(outerShape_x_cols):
        outer_shape_normal.at[subset_df.index[i], col] = x_norm[j]
    for j, col in enumerate(outerShape_y_cols):
        outer_shape_normal.at[subset_df.index[i], col] = y_norm[j]
    x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level=custom_noise_level)
    for j, col in enumerate(outerShape_x_cols):
        outer_shape_noisy.at[subset_df.index[i], col] = x_noisy[j]
    for j, col in enumerate(outerShape_y_cols):
        outer_shape_noisy.at[subset_df.index[i], col] = y_noisy[j]

# ---------------------
# Figure 1: Grid Plot for Bottom Cylinder (Normal vs Noisy)
# ---------------------
fig1, axs1 = plt.subplots(nrows, ncols, figsize=(16, 16))
for idx, (_, row_norm) in enumerate(bottom_normal.iterrows()):
    ax = axs1[idx//ncols, idx % ncols]
    x_norm = row_norm[bottom_x_cols].to_numpy(dtype=float)
    y_norm = row_norm[bottom_y_cols].to_numpy(dtype=float)
    ax.plot(x_norm, y_norm, marker='o', linestyle='-', color='blue', label='Normal')
    row_noisy = bottom_noisy.iloc[idx]
    x_noisy = row_noisy[bottom_x_cols].to_numpy(dtype=float)
    y_noisy = row_noisy[bottom_y_cols].to_numpy(dtype=float)
    ax.plot(x_noisy, y_noisy, marker='s', linestyle='--', color='red', label='Noisy')
    ax.set_title(f"Set {idx+1}")
    ax.legend(loc='upper right')
fig1.suptitle("Figure 1: Bottom Cylinder Sets (Normal vs Noisy)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------
# Figure 2: Combined Plot for Bottom Cylinder.
# ---------------------
fig2, ax2 = plt.subplots(figsize=(10, 8))
for idx, (_, row) in enumerate(bottom_normal.iterrows()):
    x_norm = row[bottom_x_cols].to_numpy(dtype=float)
    y_norm = row[bottom_y_cols].to_numpy(dtype=float)
    ax2.plot(x_norm, y_norm, marker='o', linestyle='-', color='blue', alpha=0.7, label='Normal' if idx==0 else "")
for idx, (_, row) in enumerate(bottom_noisy.iterrows()):
    x_noisy = row[bottom_x_cols].to_numpy(dtype=float)
    y_noisy = row[bottom_y_cols].to_numpy(dtype=float)
    ax2.plot(x_noisy, y_noisy, marker='s', linestyle='--', color='red', alpha=0.7, label='Noisy' if idx==0 else "")
ax2.set_title("Figure 2: Combined Bottom Cylinder Curves (Normal & Noisy)")
ax2.set_xlabel("inner_z")
ax2.set_ylabel("inner_y")
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ---------------------
# Figure 3: Grid Plot for Inner Shape (Smooth Normal vs Noisy)
# For each cell, plot the normal perfect circle and the noisy perfect circle (with sample points displayed).
# ---------------------
fig3, axs3 = plt.subplots(nrows, ncols, figsize=(16, 16))
for idx, (_, row_norm) in enumerate(inner_shape_normal.iterrows()):
    ax = axs3[idx//ncols, idx % ncols]
    x_norm_sample = row_norm[innerShape_x_cols].to_numpy(dtype=float)
    y_norm_sample = row_norm[innerShape_y_cols].to_numpy(dtype=float)
    x_norm_dense, y_norm_dense, _, _ = get_perfect_circle_line(x_norm_sample, y_norm_sample, num_points=200)
    ax.plot(x_norm_dense, y_norm_dense, color='blue', label='Normal')
    ax.scatter(x_norm_sample, y_norm_sample, color='blue', marker='o', s=40)
    row_noisy = inner_shape_noisy.iloc[idx]
    x_noisy_sample = row_noisy[innerShape_x_cols].to_numpy(dtype=float)
    y_noisy_sample = row_noisy[innerShape_y_cols].to_numpy(dtype=float)
    x_noisy_dense, y_noisy_dense, _, _ = get_perfect_circle_line(x_noisy_sample, y_noisy_sample, num_points=200)
    ax.plot(x_noisy_dense, y_noisy_dense, color='red', linestyle='--', label='Noisy')
    ax.scatter(x_noisy_sample, y_noisy_sample, color='red', marker='s', s=40)
    ax.set_title(f"Set {idx+1}")
    ax.legend(loc='upper right')
fig3.suptitle("Figure 3: Inner Shape Sets (Smooth Normal vs Noisy)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------
# Figure 4: Combined Plot for Inner Shape (Perfect Circular Splines)
# Plot normal perfect circles (blue with markers) and noisy perfect circles (red dotted).
# Ensure the legend shows one entry per type.
# ---------------------
fig4, ax4 = plt.subplots(figsize=(10, 8))
for idx, (_, row) in enumerate(inner_shape_normal.iterrows()):
    x_sample = row[innerShape_x_cols].to_numpy(dtype=float)
    y_sample = row[innerShape_y_cols].to_numpy(dtype=float)
    if idx == 0:
        plot_perfect_circle(ax4, x_sample, y_sample, label='Normal Perfect Circle', circle_color='blue')
    else:
        plot_perfect_circle(ax4, x_sample, y_sample, label='_nolegend_', circle_color='blue')
for idx, (_, row) in enumerate(inner_shape_noisy.iterrows()):
    x_sample = row[innerShape_x_cols].to_numpy(dtype=float)
    y_sample = row[innerShape_y_cols].to_numpy(dtype=float)
    x_dense, y_dense, _, _ = get_perfect_circle_line(x_sample, y_sample, num_points=200)
    if idx == 0:
        ax4.plot(x_dense, y_dense, color='red', linestyle=':', linewidth=2, label='Noisy Perfect Circle')
    else:
        ax4.plot(x_dense, y_dense, color='red', linestyle=':', linewidth=2)
ax4.set_title("Figure 4: Combined Inner Shape Curves (Perfect Circular Splines)")
ax4.set_xlabel("innerShape_x")
ax4.set_ylabel("innerShape_y")
ax4.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ---------------------
# Figure 5: Grid Plot for Outer Shape (Smooth Normal vs Noisy)
# For each cell, plot the normal perfect circle and the noisy perfect circle with sample points.
# ---------------------
fig5, axs5 = plt.subplots(nrows, ncols, figsize=(16, 16))
for idx, (_, row_norm) in enumerate(outer_shape_normal.iterrows()):
    ax = axs5[idx//ncols, idx % ncols]
    x_norm_sample = row_norm[outerShape_x_cols].to_numpy(dtype=float)
    y_norm_sample = row_norm[outerShape_y_cols].to_numpy(dtype=float)
    x_norm_dense, y_norm_dense, _, _ = get_perfect_circle_line(x_norm_sample, y_norm_sample, num_points=200)
    ax.plot(x_norm_dense, y_norm_dense, color='blue', label='Normal')
    ax.scatter(x_norm_sample, y_norm_sample, color='blue', marker='o', s=40)
    row_noisy = outer_shape_noisy.iloc[idx]
    x_noisy_sample = row_noisy[outerShape_x_cols].to_numpy(dtype=float)
    y_noisy_sample = row_noisy[outerShape_y_cols].to_numpy(dtype=float)
    x_noisy_dense, y_noisy_dense, _, _ = get_perfect_circle_line(x_noisy_sample, y_noisy_sample, num_points=200)
    ax.plot(x_noisy_dense, y_noisy_dense, color='red', linestyle='--', label='Noisy')
    ax.scatter(x_noisy_sample, y_noisy_sample, color='red', marker='s', s=40)
    ax.set_title(f"Set {idx+1}")
    ax.legend(loc='upper right')
fig5.suptitle("Figure 5: Outer Shape Sets (Smooth Normal vs Noisy)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------
# Figure 6: Combined Plot for Outer Shape (Perfect Circular Splines)
# Plot normal perfect circles (blue with markers) and noisy perfect circles (red dotted).
# Fix legend so that normal is only labeled once.
# ---------------------
fig6, ax6 = plt.subplots(figsize=(10, 8))
for idx, (_, row) in enumerate(outer_shape_normal.iterrows()):
    x_sample = row[outerShape_x_cols].to_numpy(dtype=float)
    y_sample = row[outerShape_y_cols].to_numpy(dtype=float)
    if idx == 0:
        plot_perfect_circle(ax6, x_sample, y_sample, label='Normal Perfect Circle', circle_color='blue')
    else:
        plot_perfect_circle(ax6, x_sample, y_sample, label='_nolegend_', circle_color='blue')
for idx, (_, row) in enumerate(outer_shape_noisy.iterrows()):
    x_sample = row[outerShape_x_cols].to_numpy(dtype=float)
    y_sample = row[outerShape_y_cols].to_numpy(dtype=float)
    x_dense, y_dense, _, _ = get_perfect_circle_line(x_sample, y_sample, num_points=200)
    if idx == 0:
        ax6.plot(x_dense, y_dense, color='red', linestyle=':', linewidth=2, label='Noisy Perfect Circle')
    else:
        ax6.plot(x_dense, y_dense, color='red', linestyle=':', linewidth=2)
ax6.set_title("Figure 6: Combined Outer Shape Curves (Perfect Circular Splines)")
ax6.set_xlabel("outerShape_x")
ax6.set_ylabel("outerShape_y")
ax6.legend(loc='upper right')
plt.tight_layout()
plt.show()
