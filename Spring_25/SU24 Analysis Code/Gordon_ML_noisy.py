# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:05:39 2025

@author: nerij
"""

print("Hello World")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import PostProcess_FeBio as proc
import PCA_data as pca_utils
import joblib
from Noise_Functions import add_noise, process_noisy_rows, add_noise_to_row


print("Hello World")

# Define results folder
RESULTS_FOLDER = r"C:\Users\mgordon\My Drive\a  Research\a Pelvic Floor\Inverse FEA\SU24 Analysis Code"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print(RESULTS_FOLDER)

# Load dataset
file_path = r"C:\Users\mgordon\My Drive\a  Research\a Pelvic Floor\Inverse FEA\SU24 Analysis Code\data_for_testing.csv"
df = pd.read_csv(file_path)

print(df)

# Define features and targets
features = [
    "inner_y1", "inner_y2", "inner_y3", "inner_y4", "inner_y5", "inner_y6", "inner_y7", "inner_y8", "inner_y9",
    "inner_z1", "inner_z2", "inner_z3", "inner_z4", "inner_z5", "inner_z6", "inner_z7", "inner_z8", "inner_z9",
    "outer_y1", "outer_y2", "outer_y3", "outer_y4", "outer_y5", "outer_y6", "outer_y7", "outer_y8", "outer_y9",
    "outer_z1", "outer_z2", "outer_z3", "outer_z4", "outer_z5", "outer_z6", "outer_z7", "outer_z8", "outer_z9",
    "innerShape_x1", "innerShape_x2", "innerShape_x3", "innerShape_x4", "innerShape_x5", "innerShape_x6", "innerShape_x7", "innerShape_x8", "innerShape_x9",
    "innerShape_y1", "innerShape_y2", "innerShape_y3", "innerShape_y4", "innerShape_y5", "innerShape_y6", "innerShape_y7", "innerShape_y8", "innerShape_y9",
    "outerShape_x1", "outerShape_x2", "outerShape_x3", "outerShape_x4", "outerShape_x5", "outerShape_x6", "outerShape_x7", "outerShape_x8", "outerShape_x9",
    "outerShape_y1", "outerShape_y2", "outerShape_y3", "outerShape_y4", "outerShape_y5", "outerShape_y6", "outerShape_y7", "outerShape_y8", "outerShape_y9"
]

targets = ["Part1_E", "Part3_E", "Part11_E"]

# Split into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[features], df[targets], test_size=0.2, random_state=42)

# Add noise to train data (with flag)
ADD_NOISE_TO_TRAIN = True  # Set to True to add noise, False otherwise

if ADD_NOISE_TO_TRAIN:
    noise_level_1 = 0.05
    noise_level_2 = noise_level_1

    X_train_noisy = X_train_raw.copy()

    for col in X_train_noisy.columns:
        if 'innerShape_x' in col or 'innerShape_y' in col or 'outerShape_x' in col or 'outerShape_y' in col:
            X_train_noisy[col] = add_noise_to_row(X_train_noisy[col], noise_level_1)
        if 'inner_y' in col or 'inner_z' in col or 'outer_y' in col or 'outer_z' in col:
            X_train_noisy[col] = add_noise_to_row(X_train_noisy[col], noise_level_2)

    train_file = os.path.join(RESULTS_FOLDER, "train_noisy_data.csv")
    X_train_noisy.to_csv(train_file, index=False)
else:
    train_file = os.path.join(RESULTS_FOLDER, "train_data.csv")
    X_train_raw.to_csv(train_file, index=False)


# print(X_train_raw)
# print(X_train_noisy)
test_file = os.path.join(RESULTS_FOLDER, "test_data.csv")
X_test_raw.to_csv(test_file, index=False)

# Perform PCA processing on training data
train_pca_path, pca_inner, pca_outer, pca_bottom = proc.process_features(train_file, RESULTS_FOLDER, "train", 3)

# Define PCA score columns
pca_columns = ["PC1_Inner", "PC2_Inner", "PC3_Inner", "PC1_Outer", "PC2_Outer", "PC3_Outer", "PC1_Bottom", "PC2_Bottom", "PC3_Bottom"]

# Apply PCA to training data
if ADD_NOISE_TO_TRAIN:
    inner_pca_scores_train = pca_inner.transform(X_train_noisy[[col for col in X_train_noisy.columns if 'innerShape' in col]])[:, :3]
    outer_pca_scores_train = pca_outer.transform(X_train_noisy[[col for col in X_train_noisy.columns if 'outerShape' in col]])[:, :3]
    bottom_pca_scores_train = pca_bottom.transform(X_train_noisy[[col for col in X_train_noisy.columns if 'inner_y' in col or 'inner_z' in col]])[:, :3]
    X_train = pd.DataFrame(
        np.hstack([X_train_noisy, inner_pca_scores_train, outer_pca_scores_train, bottom_pca_scores_train]),
        columns=features + pca_columns
    )
else:
    inner_pca_scores_train = pca_inner.transform(X_train_raw[[col for col in X_train_raw.columns if 'innerShape' in col]])[:, :3]
    outer_pca_scores_train = pca_outer.transform(X_train_raw[[col for col in X_train_raw.columns if 'outerShape' in col]])[:, :3]
    bottom_pca_scores_train = pca_bottom.transform(X_train_raw[[col for col in X_train_raw.columns if 'inner_y' in col or 'inner_z' in col]])[:, :3]
    X_train = pd.DataFrame(
        np.hstack([X_train_raw, inner_pca_scores_train, outer_pca_scores_train, bottom_pca_scores_train]),
        columns=features + pca_columns
    )

# Apply PCA to test data
inner_pca_scores = pca_inner.transform(X_test_raw[[col for col in X_test_raw.columns if 'innerShape' in col]])[:, :3]
outer_pca_scores = pca_outer.transform(X_test_raw[[col for col in X_test_raw.columns if 'outerShape' in col]])[:, :3]
bottom_pca_scores = pca_bottom.transform(X_test_raw[[col for col in X_test_raw.columns if 'inner_y' in col or 'inner_z' in col]])[:, :3]

X_test = pd.DataFrame(
    np.hstack([X_test_raw, inner_pca_scores, outer_pca_scores, bottom_pca_scores]),
    columns=features + pca_columns
)

X_test_pca_file = os.path.join(RESULTS_FOLDER, "test_data_pca.csv")
X_test.to_csv(X_test_pca_file, index=False)

# Add noise to test data (with flag)
ADD_NOISE_TO_TEST = True  # Set to True to add noise, False otherwise

if ADD_NOISE_TO_TEST:
    headers_to_read = ["innerShape_x", "innerShape_y", "outerShape_x", "outerShape_y", "inner_y", "inner_z", "outer_y", "outer_z"]
    noise_level_1 = 0.05
    noise_level_2 = noise_level_1

    file_path = os.path.join(RESULTS_FOLDER, "test_data.csv")

    pc_scores_IR_array, pc_scores_OR_array, pc_scores_OB_array = add_noise(file_path, headers_to_read, pca_inner, pca_outer, pca_bottom, noise_level_1, noise_level_2)

    X_test_noisy = pd.DataFrame(
        np.hstack([X_test_raw, pc_scores_IR_array, pc_scores_OR_array, pc_scores_OB_array]),
        columns=features + pca_columns
    )
else:
    X_test_noisy = X_test.copy()  # Use the original PCA-transformed test data

# Train machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict E values
y_pred = model.predict(X_test)
y_pred_noisy = model.predict(X_test_noisy)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_noisy = mean_squared_error(y_test, y_pred_noisy)
r2_noisy = r2_score(y_test, y_pred_noisy)

print(f"Without Noise - MSE: {mse}, R2: {r2}")
print(f"With Noise - MSE: {mse_noisy}, R2: {r2_noisy}")

# Save the trained model
model_file = os.path.join(RESULTS_FOLDER, "trained_model.pkl")
joblib.dump(model, model_file)

# Plot predicted vs actual values
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
    plt.savefig(os.path.join(RESULTS_FOLDER, f"pred_vs_actual_{target}.png"))
    plt.show()

# Plot percent error vs actual
for i, target in enumerate(targets):
    percent_error = np.abs((y_pred[:, i] - y_test.iloc[:, i]) / y_test.iloc[:, i]) * 100
    percent_error_noisy = np.abs((y_pred_noisy[:, i] - y_test.iloc[:, i]) / y_test.iloc[:, i]) * 100

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test.iloc[:, i], percent_error, label="Without Noise", alpha=0.6, color="blue")
    plt.scatter(y_test.iloc[:, i], percent_error_noisy, label="With Noise", alpha=0.6, color="orange")
    plt.xlabel("Actual")
    plt.ylabel("Percent Error (%)")
    plt.title(f"Percent Error vs Actual for {target}")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"percent_error_{target}.png"))
    plt.show()