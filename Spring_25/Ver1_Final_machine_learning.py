# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:43:50 2025

@author: nerij
"""

# machine_learning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===== Define Results Folder =====
RESULTS_FOLDER = r"C:/Users/nerij/Dropbox/BiomedML-main/Refined_Results"

# ===== Load Final Datasets and Targets =====
# Features (these CSVs were created by the preprocessing script)
train_features_file = os.path.join(RESULTS_FOLDER, "train_data_with_PCA.csv")
test_features_file  = os.path.join(RESULTS_FOLDER, "test_data_with_PCA.csv")
test_noisy_features_file = os.path.join(RESULTS_FOLDER, "test_noisy_data_with_PCA.csv")

X_train_final = pd.read_csv(train_features_file)
X_test_final = pd.read_csv(test_features_file)
X_test_noisy_final = pd.read_csv(test_noisy_features_file)

# Load the target variables
train_targets_file = os.path.join(RESULTS_FOLDER, "train_targets.csv")
test_targets_file = os.path.join(RESULTS_FOLDER, "test_targets.csv")
y_train = pd.read_csv(train_targets_file)
y_test = pd.read_csv(test_targets_file)

# ===== Feature Selection Options =====
# Change FEATURE_SELECTION_OPTION to one of:
# 1 - Use all coordinate data (original features) and all PCA scores.
# 2 - Use only the PCA scores (omit original coordinate data).
# 3 - Use only one PC model's scores (set SELECTED_PC_MODELS to one model e.g., ["bottom"])
# 4 - Use a pair of PC models' scores (set SELECTED_PC_MODELS to two models, e.g., ["bottom", "inner"])
FEATURE_SELECTION_OPTION = 1  # Change to 1, 2, 3, or 4 as desired
SELECTED_PC_MODELS = ["bottom"]  # For options 3 or 4. Valid values: "bottom", "inner", "outer"
NUM_PC_SCORES = 3               # Number of PCA components to use from each PC model

# Identify which columns are the original coordinates and which are PCA score columns.
# (In the preprocessing file, original coordinates were those in the list "features" and PCA scores were appended.)
# To re-create the list of original coordinate column names, define it here:
coordinate_columns = [
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

# The PCA score columns were appended in the following order:
pca_columns = [ "PC1_Bottom", "PC2_Bottom", "PC3_Bottom",
                "PC1_InnerShape", "PC2_InnerShape", "PC3_InnerShape",
                "PC1_OuterShape", "PC2_OuterShape", "PC3_OuterShape"]

# ===== Build ML Input Features Based on Selection Option =====
if FEATURE_SELECTION_OPTION == 1:
    # Use all original coordinate data and all PCA scores.
    X_train_model = X_train_final.copy()
    X_test_model = X_test_final.copy()
    X_test_noisy_model = X_test_noisy_final.copy()

elif FEATURE_SELECTION_OPTION == 2:
    # Use PCA scores only (omit original coordinates).
    X_train_model = X_train_final[pca_columns].copy()
    X_test_model = X_test_final[pca_columns].copy()
    X_test_noisy_model = X_test_noisy_final[pca_columns].copy()

elif FEATURE_SELECTION_OPTION == 3:
    # Use only one PC model's scores.
    model_sel = SELECTED_PC_MODELS[0].lower()
    if model_sel == "bottom":
        cols = [f"PC{i+1}_Bottom" for i in range(NUM_PC_SCORES)]
    elif model_sel in ["inner", "innershape"]:
        cols = [f"PC{i+1}_InnerShape" for i in range(NUM_PC_SCORES)]
    elif model_sel in ["outer", "outershape"]:
        cols = [f"PC{i+1}_OuterShape" for i in range(NUM_PC_SCORES)]
    else:
        raise ValueError("Invalid PC model selection for option 3.")
    X_train_model = X_train_final[cols].copy()
    X_test_model = X_test_final[cols].copy()
    X_test_noisy_model = X_test_noisy_final[cols].copy()

elif FEATURE_SELECTION_OPTION == 4:
    # Use a pair of PC models' scores.
    X_train_model = pd.DataFrame()
    X_test_model = pd.DataFrame()
    X_test_noisy_model = pd.DataFrame()
    for model_sel in SELECTED_PC_MODELS:
        model_sel_lower = model_sel.lower()
        if model_sel_lower == "bottom":
            df_train = X_train_final[[f"PC{i+1}_Bottom" for i in range(NUM_PC_SCORES)]].copy()
            df_test = X_test_final[[f"PC{i+1}_Bottom" for i in range(NUM_PC_SCORES)]].copy()
            df_test_noisy = X_test_noisy_final[[f"PC{i+1}_Bottom" for i in range(NUM_PC_SCORES)]].copy()
        elif model_sel_lower in ["inner", "innershape"]:
            df_train = X_train_final[[f"PC{i+1}_InnerShape" for i in range(NUM_PC_SCORES)]].copy()
            df_test = X_test_final[[f"PC{i+1}_InnerShape" for i in range(NUM_PC_SCORES)]].copy()
            df_test_noisy = X_test_noisy_final[[f"PC{i+1}_InnerShape" for i in range(NUM_PC_SCORES)]].copy()
        elif model_sel_lower in ["outer", "outershape"]:
            df_train = X_train_final[[f"PC{i+1}_OuterShape" for i in range(NUM_PC_SCORES)]].copy()
            df_test = X_test_final[[f"PC{i+1}_OuterShape" for i in range(NUM_PC_SCORES)]].copy()
            df_test_noisy = X_test_noisy_final[[f"PC{i+1}_OuterShape" for i in range(NUM_PC_SCORES)]].copy()
        else:
            raise ValueError("Unknown PC model selection in option 4.")
        X_train_model = pd.concat([X_train_model, df_train], axis=1)
        X_test_model = pd.concat([X_test_model, df_test], axis=1)
        X_test_noisy_model = pd.concat([X_test_noisy_model, df_test_noisy], axis=1)
else:
    raise ValueError("Invalid FEATURE_SELECTION_OPTION specified.")

# Optionally, you can save the final ML datasets for inspection.
model_train_file = os.path.join(RESULTS_FOLDER, "train_data_model_features.csv")
X_train_model.to_csv(model_train_file, index=False)
print("Final training data for ML model saved:", model_train_file)

# ===== Train the Model =====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_model, y_train)
print("Model training completed.")

# Predict on the test sets.
y_pred = model.predict(X_test_model)
y_pred_noisy = model.predict(X_test_noisy_model)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_noisy = mean_squared_error(y_test, y_pred_noisy)
r2_noisy = r2_score(y_test, y_pred_noisy)

print(f"Without Noise - MSE: {mse}, R2: {r2}")
print(f"With Noise    - MSE: {mse_noisy}, R2: {r2_noisy}")

# Save the trained model
import joblib
model_file = os.path.join(RESULTS_FOLDER, "trained_model.pkl")
joblib.dump(model, model_file)
print("Trained model saved:", model_file)

# ===== Plot Predicted vs Actual Values =====
targets = y_test.columns  # Assuming the target column names remain the same.
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
for i, target in enumerate(targets):
    # Compute percent error: absolute difference divided by actual value times 100.
    percent_error = np.abs((y_pred[:, i] - y_test.iloc[:, i]) / y_test.iloc[:, i]) * 100
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
