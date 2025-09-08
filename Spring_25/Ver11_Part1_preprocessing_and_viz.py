# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 23:22:01 2025

@author: nerij

Ver11_Part1_preprocessing_and_viz.py
Splits data, injects noise, visualizes, runs PCA, saves everything (including PNGs).
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# USER CONFIG
DATA_FILE = "2025_4_14_intermediate.csv"
RESULTS_FOLDER = "Refined_Results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
PLOT_FOLDER = os.path.join(RESULTS_FOLDER, "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)
RANDOM_SEED = 42

NUM_PCA = 3
DATE_PREFIX = "ver11"
NOISE_LEVEL = 0.3
USE_NOISY_TRAIN = True
USE_NOISY_TEST = True

# === NOISE MAGNITUDES (USER-SETTABLE) ===
INNER_Y_NOISE_RANGE = (-0.01, 0.01)
INNER_Z_NOISE_RANGE = (-0.01, 0.01)

ML_TARGETS = ["Part1_E", "Part3_E", "Part11_E"]
features = [
    "inner_y1", "inner_y2", "inner_y3", "inner_y4", "inner_y5",
    "inner_y6", "inner_y7", "inner_y8", "inner_y9",
    "inner_z1", "inner_z2", "inner_z3", "inner_z4", "inner_z5",
    "inner_z6", "inner_z7", "inner_z8", "inner_z9",
    "innerShape_x1", "innerShape_x2", "innerShape_x3", "innerShape_x4", "innerShape_x5",
    "innerShape_x6", "innerShape_x7", "innerShape_x8", "innerShape_x9",
    "innerShape_y1", "innerShape_y2", "innerShape_y3", "innerShape_y4", "innerShape_y5",
    "innerShape_y6", "innerShape_y7", "innerShape_y8", "innerShape_y9",
    "outerShape_x1", "outerShape_x2", "outerShape_x3", "outerShape_x4", "outerShape_x5",
    "outerShape_x6", "outerShape_x7", "outerShape_x8", "outerShape_x9",
    "outerShape_y1", "outerShape_y2", "outerShape_y3", "outerShape_y4", "outerShape_y5",
    "outerShape_y6", "outerShape_y7", "outerShape_y8", "outerShape_y9"
]

innerShape_x_cols = [f"innerShape_x{i}" for i in range(1, 10)]
innerShape_y_cols = [f"innerShape_y{i}" for i in range(1, 10)]
outerShape_x_cols = [f"outerShape_x{i}" for i in range(1, 10)]
outerShape_y_cols = [f"outerShape_y{i}" for i in range(1, 10)]
bottom_x_cols = [f'inner_z{i}' for i in range(1, 10)]
bottom_y_cols = [f'inner_y{i}' for i in range(1, 10)]

# Custom functions
import PostProcess_FeBio as proc
from Ver7_noise_and_plot_functions import add_noise_simple, create_noisy_shape, get_perfect_circle_line

# --- LOAD AND SPLIT ---
df = pd.read_csv(DATA_FILE)
X = df[features]
y = df[ML_TARGETS]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

def apply_noise(df_X, use_noise, noise_level):
    X_noisy = df_X.copy()
    if use_noise:
        for col in X_noisy.columns:
            orig = df_X[col].to_numpy()
            if col.startswith("inner_y"):
                jitter = np.random.uniform(*INNER_Y_NOISE_RANGE, size=orig.shape)
                X_noisy[col] = orig + jitter
            elif col.startswith("inner_z"):
                jitter = np.random.uniform(*INNER_Z_NOISE_RANGE, size=orig.shape)
                X_noisy[col] = orig + jitter
            else:
                noisy = add_noise_simple(orig, noise_level)
                X_noisy[col] = noisy
        for idx, row in df_X.iterrows():
            x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, noise_level)
            for j, col in enumerate(innerShape_x_cols):
                X_noisy.at[idx, col] = x_noisy[j]
            for j, col in enumerate(innerShape_y_cols):
                X_noisy.at[idx, col] = y_noisy[j]
            x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, noise_level)
            for j, col in enumerate(outerShape_x_cols):
                X_noisy.at[idx, col] = x_noisy[j]
            for j, col in enumerate(outerShape_y_cols):
                X_noisy.at[idx, col] = y_noisy[j]
    return X_noisy

X_train_noisy = apply_noise(X_train, USE_NOISY_TRAIN, NOISE_LEVEL).reset_index(drop=True)
X_test_noisy = apply_noise(X_test, USE_NOISY_TEST, NOISE_LEVEL).reset_index(drop=True)
X_train_noisy.to_csv(os.path.join(RESULTS_FOLDER, "train_noisy_data.csv"), index=False)
X_test_noisy.to_csv(os.path.join(RESULTS_FOLDER, "test_noisy_data.csv"), index=False)
y_train.to_csv(os.path.join(RESULTS_FOLDER, "train_targets.csv"), index=False)
y_test.to_csv(os.path.join(RESULTS_FOLDER, "test_targets.csv"), index=False)

# --- Data Visualization (with PNG saving) ---
num_examples = 6  # or any value you want
np.random.seed(RANDOM_SEED)
example_indices = np.random.choice(len(X_train), num_examples, replace=False)
ncols = min(num_examples, 3)
nrows = math.ceil(num_examples / ncols)

# Bottom Cylinder
fig1, axs1 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs1 = np.array(axs1).flatten()
for idx, row_idx in enumerate(example_indices):
    ax = axs1[idx]
    x_norm = X_train.iloc[row_idx][bottom_x_cols].to_numpy()
    y_norm = X_train.iloc[row_idx][bottom_y_cols].to_numpy()
    ax.plot(x_norm, y_norm, marker='o', linestyle='-', color='blue', label='Normal')
    x_noisy = X_train_noisy.iloc[row_idx][bottom_x_cols].to_numpy()
    y_noisy = X_train_noisy.iloc[row_idx][bottom_y_cols].to_numpy()
    ax.plot(x_noisy, y_noisy, marker='s', linestyle='--', color='red', label='Noisy')
    ax.set_title(f"Example {idx+1}")
    ax.set_xlabel("inner_z")
    ax.set_ylabel("inner_y")
    ax.legend()
for j in range(num_examples, len(axs1)):
    axs1[j].axis('off')
fig1.suptitle("Bottom Cylinder: Normal vs Noisy", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(os.path.join(PLOT_FOLDER, "bottom_cylinder_normal_vs_noisy.png"))
plt.show()

# Inner Shape
fig2, axs2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs2 = np.array(axs2).flatten()
for idx, row_idx in enumerate(example_indices):
    ax = axs2[idx]
    x_norm = X_train.iloc[row_idx][innerShape_x_cols].to_numpy()
    y_norm = X_train.iloc[row_idx][innerShape_y_cols].to_numpy()
    x_noisy = X_train_noisy.iloc[row_idx][innerShape_x_cols].to_numpy()
    y_noisy = X_train_noisy.iloc[row_idx][innerShape_y_cols].to_numpy()
    x_dense, y_dense, _, _ = get_perfect_circle_line(x_norm, y_norm)
    ax.plot(x_dense, y_dense, color='blue', label='Normal')
    ax.scatter(x_norm, y_norm, color='blue')
    x_noisy_dense, y_noisy_dense, _, _ = get_perfect_circle_line(x_noisy, y_noisy)
    ax.plot(x_noisy_dense, y_noisy_dense, color='red', linestyle='--', label='Noisy')
    ax.scatter(x_noisy, y_noisy, color='red')
    ax.set_title(f"Inner Shape Example {idx+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
for j in range(num_examples, len(axs2)):
    axs2[j].axis('off')
fig2.suptitle("Inner Shape: Normal vs Noisy (Perfect Circle)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(os.path.join(PLOT_FOLDER, "inner_shape_normal_vs_noisy.png"))
plt.show()

# Outer Shape
fig3, axs3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs3 = np.array(axs3).flatten()
for idx, row_idx in enumerate(example_indices):
    ax = axs3[idx]
    x_norm = X_train.iloc[row_idx][outerShape_x_cols].to_numpy()
    y_norm = X_train.iloc[row_idx][outerShape_y_cols].to_numpy()
    x_noisy = X_train_noisy.iloc[row_idx][outerShape_x_cols].to_numpy()
    y_noisy = X_train_noisy.iloc[row_idx][outerShape_y_cols].to_numpy()
    x_dense, y_dense, _, _ = get_perfect_circle_line(x_norm, y_norm)
    ax.plot(x_dense, y_dense, color='blue', label='Normal')
    ax.scatter(x_norm, y_norm, color='blue')
    x_noisy_dense, y_noisy_dense, _, _ = get_perfect_circle_line(x_noisy, y_noisy)
    ax.plot(x_noisy_dense, y_noisy_dense, color='red', linestyle='--', label='Noisy')
    ax.scatter(x_noisy, y_noisy, color='red')
    ax.set_title(f"Outer Shape Example {idx+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
for j in range(num_examples, len(axs3)):
    axs3[j].axis('off')
fig3.suptitle("Outer Shape: Normal vs Noisy (Perfect Circle)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig(os.path.join(PLOT_FOLDER, "outer_shape_normal_vs_noisy.png"))
plt.show()

# --- RUN PCA (Suppress printouts) ---
import sys
class DummyFile(object):
    def write(self, x): pass
old_stdout = sys.stdout
sys.stdout = DummyFile()
train_pca_file, pca_inner, pca_outer, pca_bottom = proc.process_features(
    os.path.join(RESULTS_FOLDER, "train_noisy_data.csv"),
    RESULTS_FOLDER, DATE_PREFIX, NUM_PCA
)
test_pca_file, _, _, _ = proc.process_features(
    os.path.join(RESULTS_FOLDER, "test_noisy_data.csv"),
    RESULTS_FOLDER, DATE_PREFIX, NUM_PCA
)
sys.stdout = old_stdout

X_train_pca = pd.read_csv(train_pca_file).reset_index(drop=True)
X_test_pca = pd.read_csv(test_pca_file).reset_index(drop=True)

# Save for part 2
X_train_noisy.to_pickle(os.path.join(RESULTS_FOLDER, "X_train_noisy.pkl"))
X_test_noisy.to_pickle(os.path.join(RESULTS_FOLDER, "X_test_noisy.pkl"))
X_train_pca.to_pickle(os.path.join(RESULTS_FOLDER, "X_train_pca.pkl"))
X_test_pca.to_pickle(os.path.join(RESULTS_FOLDER, "X_test_pca.pkl"))
y_train.to_pickle(os.path.join(RESULTS_FOLDER, "y_train.pkl"))
y_test.to_pickle(os.path.join(RESULTS_FOLDER, "y_test.pkl"))
