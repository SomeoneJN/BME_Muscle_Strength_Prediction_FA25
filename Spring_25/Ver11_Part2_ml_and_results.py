# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 23:33:27 2025

@author: nerij

Ver11_Part2_ml_and_results.py
Handles: ML, model selection, prediction/percent error plots, chained prediction, saves all plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RESULTS_FOLDER = "Refined_Results"
PLOT_FOLDER = os.path.join(RESULTS_FOLDER, "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)
NUM_PCA = 3
ML_TARGETS = ["Part1_E", "Part3_E", "Part11_E"]
RANDOM_SEED = 42

# Select MODE as before (must match part 1)
MODE = 3

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

X_train_noisy = pd.read_pickle(os.path.join(RESULTS_FOLDER, "X_train_noisy.pkl"))
X_test_noisy = pd.read_pickle(os.path.join(RESULTS_FOLDER, "X_test_noisy.pkl"))
X_train_pca = pd.read_pickle(os.path.join(RESULTS_FOLDER, "X_train_pca.pkl"))
X_test_pca = pd.read_pickle(os.path.join(RESULTS_FOLDER, "X_test_pca.pkl"))
y_train = pd.read_pickle(os.path.join(RESULTS_FOLDER, "y_train.pkl"))
y_test = pd.read_pickle(os.path.join(RESULTS_FOLDER, "y_test.pkl"))

pca_cols = [f"PC{i+1}_Bottom" for i in range(NUM_PCA)] + \
           [f"PC{i+1}_InnerShape" for i in range(NUM_PCA)] + \
           [f"PC{i+1}_OuterShape" for i in range(NUM_PCA)]

combos = []
if MODE == 1:
    combos = [
        (features, "Coordinate Only"),
    ]
elif MODE == 2:
    combos = [
        (features, "Coordinate Only"),
        (features + [f"PC{i+1}_Bottom" for i in range(NUM_PCA)], "Coordinate+BottomPCA"),
        (features + [f"PC{i+1}_InnerShape" for i in range(NUM_PCA)], "Coordinate+InnerShapePCA"),
        (features + [f"PC{i+1}_OuterShape" for i in range(NUM_PCA)], "Coordinate+OuterShapePCA"),
        (features + pca_cols, "Coordinate+AllPCA"),
    ]
elif MODE == 3:
    combos = [
        ([f"PC{i+1}_Bottom" for i in range(NUM_PCA)], "Bottom Only"),
        ([f"PC{i+1}_InnerShape" for i in range(NUM_PCA)], "InnerShape Only"),
        ([f"PC{i+1}_OuterShape" for i in range(NUM_PCA)], "OuterShape Only"),
        ([f"PC{i+1}_Bottom" for i in range(NUM_PCA)] + [f"PC{i+1}_InnerShape" for i in range(NUM_PCA)], "Bottom+Inner"),
        ([f"PC{i+1}_Bottom" for i in range(NUM_PCA)] + [f"PC{i+1}_OuterShape" for i in range(NUM_PCA)], "Bottom+Outer"),
        ([f"PC{i+1}_InnerShape" for i in range(NUM_PCA)] + [f"PC{i+1}_OuterShape" for i in range(NUM_PCA)], "Inner+Outer"),
        (pca_cols, "All PCs"),
    ]

results = {}
for target in ML_TARGETS:
    print(f"\n=== ML Model Selection for Target: {target} ===")
    best_r2 = -np.inf
    best_combo = None
    best_preds = None
    for cols, label in combos:
        if "Coordinate Only" == label:
            Xtr = X_train_noisy[cols].reset_index(drop=True)
            Xte = X_test_noisy[cols].reset_index(drop=True)
        elif "Coordinate+" in label:
            pca_cols_this = [c for c in cols if c not in features]
            Xtr = pd.concat([X_train_noisy[features], X_train_pca[pca_cols_this]], axis=1).reset_index(drop=True)
            Xte = pd.concat([X_test_noisy[features], X_test_pca[pca_cols_this]], axis=1).reset_index(drop=True)
        else:
            Xtr = X_train_pca[cols].reset_index(drop=True)
            Xte = X_test_pca[cols].reset_index(drop=True)
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(Xtr, y_train[target])
        preds = model.predict(Xte)
        r2 = r2_score(y_test[target], preds)
        mse = mean_squared_error(y_test[target], preds)
        print(f"{label:25s}: R2 = {r2:.3f} | MSE = {mse:.3f}")
        if r2 > best_r2:
            best_r2 = r2
            best_combo = label
            best_preds = preds

    # --- Save prediction vs actual (handle PCA-only case) ---
    if MODE == 3:
    # --- Find best and second-best single PCA group ---
        best_label = None
        best_preds = None
        second_label = None
        second_preds = None
        best_r2 = -np.inf
        second_r2 = -np.inf
    
        for cols, label in combos:
            if label in ["Bottom Only", "InnerShape Only", "OuterShape Only"]:
                Xtr = X_train_pca[cols].reset_index(drop=True)
                Xte = X_test_pca[cols].reset_index(drop=True)
                model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
                model.fit(Xtr, y_train[target])
                preds = model.predict(Xte)
                r2 = r2_score(y_test[target], preds)
                if r2 > best_r2:
                    second_r2 = best_r2
                    second_label = best_label
                    second_preds = best_preds
                    best_r2 = r2
                    best_label = label
                    best_preds = preds
                elif r2 > second_r2:
                    second_r2 = r2
                    second_label = label
                    second_preds = preds
    
        # --- Plot both best and second-best on the same graph ---
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_test[target], best_preds, alpha=0.7, color='tab:blue', label=f'Best: {best_label} (R2={best_r2:.3f})')
        if second_label is not None:
            plt.scatter(y_test[target], second_preds, alpha=0.7, color='tab:orange', label=f'Second: {second_label} (R2={second_r2:.3f})')
        plt.plot([y_test[target].min(), y_test[target].max()],
                 [y_test[target].min(), y_test[target].max()], 'k--', lw=2, label='Ideal')
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target}: Best & 2nd-Best Single PCA Groups")
        plt.legend()
        plt.tight_layout()
        fname = f"{target}_prediction_vs_actual_best_and_second_single_PCA.png"
        fig.savefig(os.path.join(PLOT_FOLDER, fname))
        plt.show()
    
    else:
        # Original: just best (non-PCA-only mode)
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_test[target], best_preds, alpha=0.7)
        plt.plot([y_test[target].min(), y_test[target].max()],
                 [y_test[target].min(), y_test[target].max()], 'k--', lw=2, label='Ideal')
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target}: Best ML ({best_combo})")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(PLOT_FOLDER, f"{target}_prediction_vs_actual.png"))
        plt.show()

    

    # --- Percent Error Calculation and Plots ---
    percent_error = 100 * (best_preds - y_test[target].values) / y_test[target].values

    fig = plt.figure(figsize=(7, 4))
    plt.hist(percent_error, bins=30, color='orange', edgecolor='k', alpha=0.7)
    plt.title(f"{target} Percent Error Distribution\n(Prediction vs Actual)")
    plt.xlabel("Percent Error (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_FOLDER, f"{target}_percent_error_hist.png"))
    plt.show()

    fig = plt.figure(figsize=(7, 4))
    plt.scatter(y_test[target], percent_error, alpha=0.6, color='purple')
    plt.axhline(0, color='k', linestyle='--')
    plt.title(f"{target} Percent Error vs Actual")
    plt.xlabel(f"Actual {target}")
    plt.ylabel("Percent Error (%)")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_FOLDER, f"{target}_percent_error_vs_actual.png"))
    plt.show()

    results[target] = (best_r2, best_combo, best_preds)

# === BEST PCA SETS FOR EACH TARGET (single/double/all) ===
def safe_fmt(val, width=16):
    return f"{val:{width}s}" if isinstance(val, str) and val is not None else "N/A".ljust(width)
def safe_flt(val, fmt=".3f"):
    return f"{val:{fmt}}" if isinstance(val, (float, int)) and val is not None else "N/A"

print("\n=== BEST PCA SETS FOR EACH TARGET ===")
for t in ML_TARGETS:
    best_single = None
    best_double = None
    best_all = None
    best_single_r2 = -np.inf
    best_double_r2 = -np.inf
    all_3_PCA_s_r2 = None
    all_mse = None
    single_label = ""
    double_label = ""
    single_mse = None
    double_mse = None
    for cols, label in combos:
        if label not in ["Bottom Only", "InnerShape Only", "OuterShape Only",
                         "Bottom+Inner", "Bottom+Outer", "Inner+Outer", "All PCs"]:
            continue
        Xtr = X_train_pca[cols].reset_index(drop=True)
        Xte = X_test_pca[cols].reset_index(drop=True)
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(Xtr, y_train[t])
        preds = model.predict(Xte)
        r2 = r2_score(y_test[t], preds)
        mse = mean_squared_error(y_test[t], preds)
        if label in ["Bottom Only", "InnerShape Only", "OuterShape Only"]:
            if r2 > best_single_r2:
                best_single_r2 = r2
                best_single = label
                single_mse = mse
        if label in ["Bottom+Inner", "Bottom+Outer", "Inner+Outer"]:
            if r2 > best_double_r2:
                best_double_r2 = r2
                best_double = label
                double_mse = mse
        if label == "All PCs":
            all_3_PCA_s_r2 = r2
            all_mse = mse

    print(f"\n{t}:")
    print(f"  Best Single PCA: {safe_fmt(best_single)} \n  | R2 = {safe_flt(best_single_r2)} | MSE = {safe_flt(single_mse, '.5f')} |")
    print(f"  Best Double PCA: {safe_fmt(best_double)} \n  | R2 = {safe_flt(best_double_r2)} | MSE = {safe_flt(double_mse, '.5f')} |")
    print(f"  All  groups PCA: {'All PCs':16s}   \n  | R2 = {safe_flt(all_3_PCA_s_r2)} | MSE = {safe_flt(all_mse, '.5f')} |")

# ==== CHAINED PREDICTION FOR Part1_E (robust, with percent error, PNG saving) ====
print("\n=== FINAL CHAINED PREDICTION FOR Part1_E (with P3, P11 as features) ===")

def get_best_cols_and_matrix(target_label):
    if target_label not in results:
        raise KeyError(f"{target_label} not found in results! keys: {list(results.keys())}")
    combo_label = results[target_label][1]
    for cols, label in combos:
        if label == combo_label:
            if "Coordinate Only" == label:
                return cols, "coord"
            elif "Coordinate+" in label:
                return cols, "coord+pca"
            else:
                return cols, "pca"
    return pca_cols, "pca"

def build_matrix(X_noisy, X_pca, cols, mat):
    if mat == "coord":
        return X_noisy[cols].reset_index(drop=True)
    elif mat == "coord+pca":
        pca_cols_this = [c for c in cols if c not in features]
        return pd.concat([X_noisy[features], X_pca[pca_cols_this]], axis=1).reset_index(drop=True)
    else:
        return X_pca[cols].reset_index(drop=True)

cols3, mat3 = get_best_cols_and_matrix("Part3_E")
cols11, mat11 = get_best_cols_and_matrix("Part11_E")
cols1, mat1 = get_best_cols_and_matrix("Part1_E")

X_test_3 = build_matrix(X_test_noisy, X_test_pca, cols3, mat3)
X_train_3 = build_matrix(X_train_noisy, X_train_pca, cols3, mat3)
X_test_11 = build_matrix(X_test_noisy, X_test_pca, cols11, mat11)
X_train_11 = build_matrix(X_train_noisy, X_train_pca, cols11, mat11)
X_test_1 = build_matrix(X_test_noisy, X_test_pca, cols1, mat1)
X_train_1 = build_matrix(X_train_noisy, X_train_pca, cols1, mat1)

model_3 = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_3.fit(X_train_3, y_train["Part3_E"])
part3_pred = model_3.predict(X_test_3)

model_11 = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_11.fit(X_train_11, y_train["Part11_E"])
part11_pred = model_11.predict(X_test_11)

# With TRUE values for P3/P11
X_test_1_true = X_test_1.copy()
X_test_1_true["Part3_E"] = y_test["Part3_E"].values
X_test_1_true["Part11_E"] = y_test["Part11_E"].values

X_train_1_true = X_train_1.copy()
X_train_1_true["Part3_E"] = y_train["Part3_E"].values
X_train_1_true["Part11_E"] = y_train["Part11_E"].values

model_part1_true = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_part1_true.fit(X_train_1_true, y_train["Part1_E"])
part1_pred_true = model_part1_true.predict(X_test_1_true)

# With PREDICTED values for P3/P11
X_test_1_pred = X_test_1.copy()
X_test_1_pred["Part3_E"] = part3_pred
X_test_1_pred["Part11_E"] = part11_pred

model_part1_pred = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
model_part1_pred.fit(X_train_1_true, y_train["Part1_E"])  # always train with true for fairness
part1_pred_pred = model_part1_pred.predict(X_test_1_pred)

fig = plt.figure(figsize=(7, 5))
plt.scatter(y_test["Part1_E"], part1_pred_true, color='blue', alpha=0.6, label='True P3, P11')
plt.scatter(y_test["Part1_E"], part1_pred_pred, color='red', alpha=0.6, label='Predicted P3, P11')
plt.plot([y_test["Part1_E"].min(), y_test["Part1_E"].max()],
         [y_test["Part1_E"].min(), y_test["Part1_E"].max()], 'k--', lw=2, label='Ideal')
plt.xlabel("Actual Part1_E")
plt.ylabel("Predicted Part1_E")
plt.title("Part1_E: True vs. Predicted P3/P11 as Extra Features")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOT_FOLDER, "Part1_E_chained_pred_vs_actual.png"))
plt.show()

percent_error_true = 100 * (part1_pred_true - y_test["Part1_E"].values) / y_test["Part1_E"].values
percent_error_pred = 100 * (part1_pred_pred - y_test["Part1_E"].values) / y_test["Part1_E"].values

fig = plt.figure(figsize=(7, 4))
plt.hist(percent_error_true, bins=30, color='skyblue', edgecolor='k', alpha=0.7, label='True P3/P11')
plt.hist(percent_error_pred, bins=30, color='salmon', edgecolor='k', alpha=0.7, label='Predicted P3/P11')
plt.title("Part1_E Chained Prediction Percent Error Distribution")
plt.xlabel("Percent Error (%)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOT_FOLDER, "Part1_E_chained_percent_error_hist.png"))
plt.show()

fig = plt.figure(figsize=(7, 4))
plt.scatter(y_test["Part1_E"], percent_error_true, alpha=0.6, color='blue', label='True P3/P11')
plt.scatter(y_test["Part1_E"], percent_error_pred, alpha=0.6, color='red', label='Predicted P3/P11')
plt.axhline(0, color='k', linestyle='--')
plt.title("Part1_E Chained Prediction Percent Error vs Actual")
plt.xlabel("Actual Part1_E")
plt.ylabel("Percent Error (%)")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOT_FOLDER, "Part1_E_chained_percent_error_vs_actual.png"))
plt.show()

r2_true = r2_score(y_test["Part1_E"], part1_pred_true)
r2_pred = r2_score(y_test["Part1_E"], part1_pred_pred)
print(f"R2 with TRUE Part3_E, Part11_E: {r2_true:.3f}")
print(f"R2 with PREDICTED Part3_E, Part11_E: {r2_pred:.3f}")
