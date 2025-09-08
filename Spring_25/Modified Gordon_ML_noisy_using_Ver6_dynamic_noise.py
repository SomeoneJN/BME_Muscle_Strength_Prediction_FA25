# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 02:02:19 2025

@author: nerij
"""

# Modified Gordon_ML_noisy_using_Ver6_dynamic_noise_with_rows_and_percent_error.py
# This script uses noise functions from Ver6_noise_and_plot_functions and computes
# a dynamic noise level for inner_z (based on the standard deviation of the first N rows).
# It also adds an option to limit the number of rows used in the dataset (default is all rows)
# and creates an additional plot showing the percent error of the predictions compared to the actual values.
#
# The features considered are:
#  - inner_y and inner_z (numerical columns, with noise added using add_noise_simple)
#  - innerShape and outerShape (noise added using create_noisy_shape)
#
# The dataset is split into train and test sets, raw and noisy versions are saved to CSV.
# The training set (raw or noisy) is then processed by process_features (from PostProcess_FeBio)
# to append PCA scores, and the same PCA transformations are applied to the test data.
#
# Finally, a RandomForestRegressor is trained, predictions are made on both clean and noisy test sets,
# and performance metrics as well as percent error plots are generated.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import PostProcess_FeBio as proc
import joblib

# Import noise functions from Ver6_noise_and_plot_functions.py
from Ver6_noise_and_plot_functions import add_noise_simple, create_noisy_shape

# ===== Option Flags and Parameters =====
USE_NOISY_TRAIN = True   # Set True to add noise to training data
USE_NOISY_TEST  = True   # Set True to add noise to test data
preset_noise_level = 0.1  # Preset noise level for inner_y, innerShape and outerShape
# Option to select number of rows to use from the dataset (set to None for all rows)
rows_to_use = 2500 # e.g., set rows_to_use = 5000 to use only the first 5000 rows

# ===== Define Results Folder =====
RESULTS_FOLDER = r"C:\Users\nerij\Dropbox\FE\Refined_Results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print("Results will be saved to:", RESULTS_FOLDER)

# ===== Load Dataset =====
file_path = r"C:/Users/nerij/Dropbox/FE/2025_3_3_intermediate.csv"
df = pd.read_csv(file_path)
if rows_to_use is not None:
    df = df.iloc[:rows_to_use]
print("Dataset loaded, shape:", df.shape)

# ===== Compute Dynamic Noise Level for inner_z =====
inner_z_cols = [f"inner_z{i}" for i in range(1, 10)]
dynamic_noise_inner_z = round(df[inner_z_cols].iloc[:1000].std(axis=0).mean(), 2)
print("Dynamic noise level for inner_z computed as:", dynamic_noise_inner_z)

# ===== Define Features and Targets =====
features = [
    # inner_y and inner_z columns
    "inner_y1", "inner_y2", "inner_y3", "inner_y4", "inner_y5",
    "inner_y6", "inner_y7", "inner_y8", "inner_y9",
    "inner_z1", "inner_z2", "inner_z3", "inner_z4", "inner_z5",
    "inner_z6", "inner_z7", "inner_z8", "inner_z9",
    # innerShape columns
    "innerShape_x1", "innerShape_x2", "innerShape_x3", "innerShape_x4", "innerShape_x5",
    "innerShape_x6", "innerShape_x7", "innerShape_x8", "innerShape_x9",
    "innerShape_y1", "innerShape_y2", "innerShape_y3", "innerShape_y4", "innerShape_y5",
    "innerShape_y6", "innerShape_y7", "innerShape_y8", "innerShape_y9",
    # outerShape columns
    "outerShape_x1", "outerShape_x2", "outerShape_x3", "outerShape_x4", "outerShape_x5",
    "outerShape_x6", "outerShape_x7", "outerShape_x8", "outerShape_x9",
    "outerShape_y1", "outerShape_y2", "outerShape_y3", "outerShape_y4", "outerShape_y5",
    "outerShape_y6", "outerShape_y7", "outerShape_y8", "outerShape_y9"
]
targets = ["Part1_E", "Part3_E", "Part11_E"]

# Define shape column groups for processing with create_noisy_shape
innerShape_x_cols = [f"innerShape_x{i}" for i in range(1, 10)]
innerShape_y_cols = [f"innerShape_y{i}" for i in range(1, 10)]
outerShape_x_cols = [f"outerShape_x{i}" for i in range(1, 10)]
outerShape_y_cols = [f"outerShape_y{i}" for i in range(1, 10)]

# ===== Split Data into Train and Test =====
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df[features], df[targets], test_size=0.2, random_state=42
)

# Save raw splits to CSV
train_raw_file = os.path.join(RESULTS_FOLDER, "train_data.csv")
test_raw_file  = os.path.join(RESULTS_FOLDER, "test_data.csv")
X_train_raw.to_csv(train_raw_file, index=False)
X_test_raw.to_csv(test_raw_file, index=False)
print("Train and test splits saved (raw).")

# ===== Add Noise to Training Data =====
X_train_noisy = X_train_raw.copy()
if USE_NOISY_TRAIN:
    # For inner_y, use preset noise level; for inner_z, use dynamic noise level.
    for col in X_train_noisy.columns:
        if "inner_y" in col:
            X_train_noisy[col] = add_noise_simple(X_train_raw[col].to_numpy(), preset_noise_level)
        if "inner_z" in col:
            X_train_noisy[col] = add_noise_simple(X_train_raw[col].to_numpy(), dynamic_noise_inner_z)
    
    # For innerShape and outerShape groups, process each row using create_noisy_shape.
    for idx, row in X_train_raw.iterrows():
        x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, preset_noise_level)
        for j, col in enumerate(innerShape_x_cols):
            X_train_noisy.at[idx, col] = x_noisy[j]
        for j, col in enumerate(innerShape_y_cols):
            X_train_noisy.at[idx, col] = y_noisy[j]
    for idx, row in X_train_raw.iterrows():
        x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, preset_noise_level)
        for j, col in enumerate(outerShape_x_cols):
            X_train_noisy.at[idx, col] = x_noisy[j]
        for j, col in enumerate(outerShape_y_cols):
            X_train_noisy.at[idx, col] = y_noisy[j]
    
    train_noisy_file = os.path.join(RESULTS_FOLDER, "train_noisy_data.csv")
    X_train_noisy.to_csv(train_noisy_file, index=False)
    print("Noisy training data saved.")
else:
    train_noisy_file = train_raw_file

# ===== Process Training Features with PCA =====
# process_features groups columns into three blocks (bottom: inner_y+inner_z, innerShape, outerShape)
modified_train_file, pca_inner, pca_outer, pca_bottom = proc.process_features(
    train_noisy_file, RESULTS_FOLDER, "train", 3
)
print("Modified training file created with PCA scores:", modified_train_file)

# Define the names for the new PCA score columns.
# Note: In our modified process_features the PCA models correspond to:
#   - bottom: from inner_y and inner_z,
#   - innerShape, and
#   - outerShape.
pca_columns = [ "PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
                "PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape",
                "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"]

# ===== Transform Training Data using PCA Models =====
# Group definitions for PCA transformation:
inner_shape_cols = [col for col in X_train_noisy.columns if "innerShape" in col]
outer_shape_cols = [col for col in X_train_noisy.columns if "outerShape" in col]
bottom_cols      = [col for col in X_train_noisy.columns if ("inner_y" in col or "inner_z" in col)]

inner_pca_scores_train = pca_inner.transform(X_train_noisy[inner_shape_cols])[:, :3]
outer_pca_scores_train = pca_outer.transform(X_train_noisy[outer_shape_cols])[:, :3]
bottom_pca_scores_train = pca_bottom.transform(X_train_noisy[bottom_cols])[:, :3]

# Combine original features with the PCA scores.
X_train_final = pd.DataFrame(
    np.hstack([X_train_noisy,
               bottom_pca_scores_train,
               inner_pca_scores_train,
               outer_pca_scores_train]),
    columns=features + pca_columns
)

final_train_file = os.path.join(RESULTS_FOLDER, "train_data_with_PCA.csv")
X_train_final.to_csv(final_train_file, index=False)
print("Final training data (with PCA scores) saved:", final_train_file)

# ===== Process Test Data =====
X_test_noisy = X_test_raw.copy()
if USE_NOISY_TEST:
    for col in X_test_noisy.columns:
        if "inner_y" in col:
            X_test_noisy[col] = add_noise_simple(X_test_raw[col].to_numpy(), preset_noise_level)
        if "inner_z" in col:
            X_test_noisy[col] = add_noise_simple(X_test_raw[col].to_numpy(), dynamic_noise_inner_z)
    for idx, row in X_test_raw.iterrows():
        x_noisy, y_noisy = create_noisy_shape(row, innerShape_x_cols, innerShape_y_cols, preset_noise_level)
        for j, col in enumerate(innerShape_x_cols):
            X_test_noisy.at[idx, col] = x_noisy[j]
        for j, col in enumerate(innerShape_y_cols):
            X_test_noisy.at[idx, col] = y_noisy[j]
    for idx, row in X_test_raw.iterrows():
        x_noisy, y_noisy = create_noisy_shape(row, outerShape_x_cols, outerShape_y_cols, preset_noise_level)
        for j, col in enumerate(outerShape_x_cols):
            X_test_noisy.at[idx, col] = x_noisy[j]
        for j, col in enumerate(outerShape_y_cols):
            X_test_noisy.at[idx, col] = y_noisy[j]
    
    test_noisy_file = os.path.join(RESULTS_FOLDER, "test_noisy_data.csv")
    X_test_noisy.to_csv(test_noisy_file, index=False)
    print("Noisy test data saved.")
else:
    test_noisy_file = test_raw_file

# ===== Apply PCA Transformation to Test Data =====
inner_shape_cols_test = [col for col in X_test_raw.columns if "innerShape" in col]
outer_shape_cols_test = [col for col in X_test_raw.columns if "outerShape" in col]
bottom_cols_test      = [col for col in X_test_raw.columns if ("inner_y" in col or "inner_z" in col)]

inner_pca_scores_test = pca_inner.transform(X_test_raw[inner_shape_cols_test])[:, :3]
outer_pca_scores_test = pca_outer.transform(X_test_raw[outer_shape_cols_test])[:, :3]
bottom_pca_scores_test = pca_bottom.transform(X_test_raw[bottom_cols_test])[:, :3]

X_test_final = pd.DataFrame(
    np.hstack([X_test_raw,
               bottom_pca_scores_test,
               inner_pca_scores_test,
               outer_pca_scores_test]),
    columns=features + pca_columns
)

inner_pca_scores_test_noisy = pca_inner.transform(X_test_noisy[inner_shape_cols_test])[:, :3]
outer_pca_scores_test_noisy = pca_outer.transform(X_test_noisy[outer_shape_cols_test])[:, :3]
bottom_pca_scores_test_noisy = pca_bottom.transform(X_test_noisy[bottom_cols_test])[:, :3]

X_test_noisy_final = pd.DataFrame(
    np.hstack([X_test_noisy,
               bottom_pca_scores_test_noisy,
               inner_pca_scores_test_noisy,
               outer_pca_scores_test_noisy]),
    columns=features + pca_columns
)

final_test_file = os.path.join(RESULTS_FOLDER, "test_data_with_PCA.csv")
final_test_noisy_file = os.path.join(RESULTS_FOLDER, "test_noisy_data_with_PCA.csv")
X_test_final.to_csv(final_test_file, index=False)
X_test_noisy_final.to_csv(final_test_noisy_file, index=False)
print("Final test data (with PCA scores) saved:", final_test_file)
print("Final noisy test data (with PCA scores) saved:", final_test_noisy_file)

# ===== Train Machine Learning Model =====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)
print("Model training completed.")

# Predict on test data (both clean and noisy)
y_pred = model.predict(X_test_final)
y_pred_noisy = model.predict(X_test_noisy_final)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_noisy = mean_squared_error(y_test, y_pred_noisy)
r2_noisy = r2_score(y_test, y_pred_noisy)
print(f"Without Noise - MSE: {mse}, R2: {r2}")
print(f"With Noise    - MSE: {mse_noisy}, R2: {r2_noisy}")

# Save the trained model
model_file = os.path.join(RESULTS_FOLDER, "trained_model.pkl")
joblib.dump(model, model_file)
print("Trained model saved:", model_file)

# ===== Plot Predicted vs Actual Values =====
for i, target in enumerate(targets):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], label="Without Noise", alpha=0.6, color="blue")
    plt.scatter(y_test.iloc[:, i], y_pred_noisy[:, i], label="With Noise", alpha=0.6, color="orange")
    plt.plot([min(y_test.iloc[:, i]), max(y_test.iloc[:, i])],
             [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual for {target}")
    plt.legend()
    plot_file = os.path.join(RESULTS_FOLDER, f"pred_vs_actual_{target}.png")
    plt.savefig(plot_file)
    plt.show()
    print("Plot saved:", plot_file)

# ===== Plot Percent Error vs Actual =====
# Percent error is computed as the absolute difference divided by actual, times 100.
for i, target in enumerate(targets):
    # For predictions without noise
    percent_error = np.abs((y_pred[:, i] - y_test.iloc[:, i]) / y_test.iloc[:, i]) * 100
    # For noisy predictions
    percent_error_noisy = np.abs((y_pred_noisy[:, i] - y_test.iloc[:, i]) / y_test.iloc[:, i]) * 100
    
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test.iloc[:, i], percent_error, label="Percent Error (No Noise)", alpha=0.6, color="blue")
    plt.scatter(y_test.iloc[:, i], percent_error_noisy, label="Percent Error (With Noise)", alpha=0.6, color="orange")
    plt.xlabel("Actual")
    plt.ylabel("Percent Error (%)")
    plt.title(f"Percent Error vs Actual for {target}")
    plt.legend()
    plot_file = os.path.join(RESULTS_FOLDER, f"percent_error_{target}.png")
    plt.savefig(plot_file)
    plt.show()
    print("Percent error plot saved:", plot_file)
