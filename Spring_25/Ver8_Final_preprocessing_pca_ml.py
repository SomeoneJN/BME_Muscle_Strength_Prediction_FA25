# -*- coding: utf-8 -*-
"""
Created on [date]
@author: nerij

Script: Ver8_Final_preprocessing_pca_ml.py

Workflow:
1. Split data, add optional noise, save splits.
2. Visualize original vs noisy data (cylinder & shapes).
3. Run PCA for bottom/inner/outer groups.
4. Train/test ML models using PC scores.
5. Plot predictions vs actual for both original and noisy features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === USER CONFIGURATION ===
DATA_FILE = "2025_4_14_intermediate.csv"
RESULTS_FOLDER = "Refined_Results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# NOISE CONTROL
USE_NOISY_TRAIN = True
USE_NOISY_TEST = True
NOISE_LEVEL = 0.3  # Can set between 0.25 and 0.5
RANDOM_SEED = 42

# PCA CONFIG
NUM_PCA = 3
DATE_PREFIX = "ver8"

# ML OPTIONS
ML_TARGETS = ["Part1_E", "Part3_E", "Part11_E"]

# === IMPORT CUSTOM FUNCTIONS ===
import PostProcess_FeBio as proc
from Ver7_noise_and_plot_functions import add_noise_simple, create_noisy_shape, get_perfect_circle_line

# ==== DEFINE COLUMNS ====
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
targets = ML_TARGETS

innerShape_x_cols = [f"innerShape_x{i}" for i in range(1, 10)]
innerShape_y_cols = [f"innerShape_y{i}" for i in range(1, 10)]
outerShape_x_cols = [f"outerShape_x{i}" for i in range(1, 10)]
outerShape_y_cols = [f"outerShape_y{i}" for i in range(1, 10)]
bottom_x_cols = [f'inner_z{i}' for i in range(1, 10)]
bottom_y_cols = [f'inner_y{i}' for i in range(1, 10)]

# ==== LOAD DATA ====
df = pd.read_csv(DATA_FILE)
print(f"Loaded data: {df.shape}")

# ==== SPLIT DATA ====
X = df[features]
y = df[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# ==== NOISE INJECTION ====
def apply_noise(df_X, use_noise, noise_level):
    X_noisy = df_X.copy()
    if use_noise:
        for col in X_noisy.columns:
            orig = df_X[col].to_numpy()
            if col.startswith("inner_y"):
                jitter = np.random.uniform(-0.015, 0.015, size=orig.shape)
                X_noisy[col] = orig + jitter
            elif col.startswith("inner_z"):
                jitter = np.random.uniform(-0.015, 0.015, size=orig.shape)
                X_noisy[col] = orig + jitter
            else:
                noisy = add_noise_simple(orig, noise_level)
                max_dev = np.clip(noisy - orig, -0.5, 0.5)
                X_noisy[col] = orig + max_dev
        # Noisy shapes (still uses noise_level)
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

X_train_noisy = apply_noise(X_train, USE_NOISY_TRAIN, NOISE_LEVEL)
X_test_noisy = apply_noise(X_test, USE_NOISY_TEST, NOISE_LEVEL)

# ==== SAVE SPLITS ====
X_train_noisy.to_csv(os.path.join(RESULTS_FOLDER, "train_noisy_data.csv"), index=False)
X_test_noisy.to_csv(os.path.join(RESULTS_FOLDER, "test_noisy_data.csv"), index=False)
y_train.to_csv(os.path.join(RESULTS_FOLDER, "train_targets.csv"), index=False)
y_test.to_csv(os.path.join(RESULTS_FOLDER, "test_targets.csv"), index=False)

# ===========================
# ==== DATA VISUALIZATION ===
# ===========================
num_examples = 6
np.random.seed(RANDOM_SEED)
example_indices = np.random.choice(len(X_train), num_examples, replace=False)

# ---- Plot: Bottom Cylinder ----
fig1, axs1 = plt.subplots(2, 3, figsize=(15, 8))
axs1 = axs1.flatten()
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
fig1.suptitle("Bottom Cylinder: Normal vs Noisy", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---- Plot: Inner Shape (Perfect Circle) ----
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
axs2 = axs2.flatten()
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
fig2.suptitle("Inner Shape: Normal vs Noisy (Perfect Circle)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---- Plot: Outer Shape (Perfect Circle) ----
fig3, axs3 = plt.subplots(2, 3, figsize=(15, 8))
axs3 = axs3.flatten()
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
fig3.suptitle("Outer Shape: Normal vs Noisy (Perfect Circle)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ==========================
# ==== RUN PCA (GROUPS) ====
# ==========================
train_pca_file, pca_inner, pca_outer, pca_bottom = proc.process_features(
    os.path.join(RESULTS_FOLDER, "train_noisy_data.csv"),
    RESULTS_FOLDER, DATE_PREFIX, NUM_PCA
)
test_pca_file, _, _, _ = proc.process_features(
    os.path.join(RESULTS_FOLDER, "test_noisy_data.csv"),
    RESULTS_FOLDER, DATE_PREFIX, NUM_PCA
)

# ==== LOAD PC SCORES ====
X_train_pca = pd.read_csv(train_pca_file)
X_test_pca = pd.read_csv(test_pca_file)

pca_cols = [f"PC{i+1}_Bottom" for i in range(NUM_PCA)] + \
           [f"PC{i+1}_InnerShape" for i in range(NUM_PCA)] + \
           [f"PC{i+1}_OuterShape" for i in range(NUM_PCA)]

# ==== ML SEARCH AND PREDICTION VS ACTUAL PLOTS ====
results = {}
combos = [
    (["PC1_Bottom", "PC2_Bottom", "PC3_Bottom"], "Bottom Only"),
    (["PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape"], "InnerShape Only"),
    (["PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"], "OuterShape Only"),
    (["PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
      "PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape"], "Bottom+Inner"),
    (["PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
      "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"], "Bottom+Outer"),
    (["PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape",
      "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"], "Inner+Outer"),
    (pca_cols, "All PCs")
]

for target in ML_TARGETS:
    print(f"\n=== ML Model Selection for Target: {target} ===")
    best_r2 = -np.inf
    best_combo = None
    best_preds = None

    for cols, label in combos:
        Xtr = X_train_pca[cols]
        Xte = X_test_pca[cols]
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(Xtr, y_train[target])
        preds = model.predict(Xte)
        r2 = r2_score(y_test[target], preds)
        print(f"{label:16s}: R2 = {r2:.3f}")
        if r2 > best_r2:
            best_r2 = r2
            best_combo = label
            best_preds = preds
    results[target] = (best_r2, best_combo, best_preds)

print("\n=== BEST PCA SET FOR EACH TARGET ===")
for t in ML_TARGETS:
    print(f"{t}: Best R2 = {results[t][0]:.3f}  |  Best PC group: {results[t][1]}")

# ==== PREDICTIONS VS ACTUAL (Noisy) ====
for idx, target in enumerate(ML_TARGETS):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test[target], results[target][2], alpha=0.7, label='Noisy Prediction')
    plt.plot([y_test[target].min(), y_test[target].max()],
             [y_test[target].min(), y_test[target].max()], 'k--', lw=2, label='Ideal')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Noisy Prediction vs Actual for {target}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==== PREDICTIONS VS ACTUAL (Original, No Noise) ====
# Re-run ML using original (non-noisy) features
train_pca_file_orig, _, _, _ = proc.process_features(
    os.path.join(RESULTS_FOLDER, "train_data.csv"),
    RESULTS_FOLDER, DATE_PREFIX+"_orig", NUM_PCA
)
test_pca_file_orig, _, _, _ = proc.process_features(
    os.path.join(RESULTS_FOLDER, "test_data.csv"),
    RESULTS_FOLDER, DATE_PREFIX+"_orig", NUM_PCA
)
X_train_pca_orig = pd.read_csv(train_pca_file_orig)
X_test_pca_orig = pd.read_csv(test_pca_file_orig)

for idx, target in enumerate(ML_TARGETS):
    # Use same best PC group as found for noisy
    best_cols = []
    for cols, label in combos:
        if label == results[target][1]:
            best_cols = cols
            break
    Xtr = X_train_pca_orig[best_cols]
    Xte = X_test_pca_orig[best_cols]
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(Xtr, y_train[target])
    preds = model.predict(Xte)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test[target], preds, alpha=0.7, label='Original Prediction')
    plt.plot([y_test[target].min(), y_test[target].max()],
             [y_test[target].min(), y_test[target].max()], 'k--', lw=2, label='Ideal')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Original (No Noise) Prediction vs Actual for {target}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ==== FINAL ML PREDICTION WITH CHAINED TARGETS ====
print("\n=== FINAL CHAINED PREDICTION FOR Part1_E ===")

def get_best_cols(target_label):
    for cols, label in combos:
        if label == results[target_label][1]:
            return cols
    return pca_cols

# Step 1: Predict Part3_E and Part11_E on test data using their best PC groups
cols3 = get_best_cols("Part3_E")
cols11 = get_best_cols("Part11_E")
X_test_pca_for3 = X_test_pca[cols3]
X_test_pca_for11 = X_test_pca[cols11]

model_3 = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_3.fit(X_train_pca[cols3], y_train["Part3_E"])
part3_pred = model_3.predict(X_test_pca_for3)

model_11 = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_11.fit(X_train_pca[cols11], y_train["Part11_E"])
part11_pred = model_11.predict(X_test_pca_for11)

# Step 2: Get best PC group for Part1_E
cols1 = get_best_cols("Part1_E")

# Prepare features for chained prediction
X_test_part1_true = X_test_pca[cols1].copy()
X_test_part1_true["Part3_E"] = y_test["Part3_E"].values
X_test_part1_true["Part11_E"] = y_test["Part11_E"].values

X_train_part1_true = X_train_pca[cols1].copy()
X_train_part1_true["Part3_E"] = y_train["Part3_E"].values
X_train_part1_true["Part11_E"] = y_train["Part11_E"].values

# With TRUE values for P3/P11
model_part1_true = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_part1_true.fit(X_train_part1_true, y_train["Part1_E"])
part1_pred_true = model_part1_true.predict(X_test_part1_true)

# With PREDICTED values for P3/P11
X_test_part1_pred = X_test_pca[cols1].copy()
X_test_part1_pred["Part3_E"] = part3_pred
X_test_part1_pred["Part11_E"] = part11_pred

X_train_part1_pred = X_train_pca[cols1].copy()
X_train_part1_pred["Part3_E"] = y_train["Part3_E"].values
X_train_part1_pred["Part11_E"] = y_train["Part11_E"].values

model_part1_pred = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_part1_pred.fit(X_train_part1_pred, y_train["Part1_E"])
part1_pred_pred = model_part1_pred.predict(X_test_part1_pred)

# ==== PLOT & R2 FOR CHAINED PREDICTION OF Part1_E ====
plt.figure(figsize=(7, 5))
plt.scatter(y_test["Part1_E"], part1_pred_true, color='blue', alpha=0.6, label='True P3, P11')
plt.scatter(y_test["Part1_E"], part1_pred_pred, color='red', alpha=0.6, label='Predicted P3, P11')
plt.plot([y_test["Part1_E"].min(), y_test["Part1_E"].max()],
         [y_test["Part1_E"].min(), y_test["Part1_E"].max()], 'k--', lw=2, label='Ideal')
plt.xlabel("Actual Part1_E")
plt.ylabel("Predicted Part1_E")
plt.title("Part1_E: True vs. Predicted P3/P11 as Extra Features")
plt.legend()
plt.tight_layout()
plt.show()

r2_true = r2_score(y_test["Part1_E"], part1_pred_true)
r2_pred = r2_score(y_test["Part1_E"], part1_pred_pred)
print(f"R2 with TRUE Part3_E, Part11_E: {r2_true:.3f}")
print(f"R2 with PREDICTED Part3_E, Part11_E: {r2_pred:.3f}")

