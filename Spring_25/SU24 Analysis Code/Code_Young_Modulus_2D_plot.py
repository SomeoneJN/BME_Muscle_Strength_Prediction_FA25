import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def plot_actual_vs_predicted(y_test, y_pred, title_suffix=""):
    """Plot actual vs predicted Young's modulus values."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))
    colors = ['blue', 'green', 'orange']

    for i, color in enumerate(colors):
        min_val, max_val = min(y_test[:, i].min(), y_pred[:, i].min()), max(y_test[:, i].max(), y_pred[:, i].max())
        
        axs[i].scatter(y_test[:, i], y_pred[:, i], color=color, alpha=0.6)
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle="--")
        axs[i].set_xlabel(f'Actual Youngs_Modulus{i+1}')
        axs[i].set_ylabel(f'Predicted Youngs_Modulus{i+1}')
        axs[i].set_title(f'Youngs_Modulus{i+1}: Actual vs Predicted {title_suffix}')
        axs[i].set_xlim(min_val, max_val)
        axs[i].set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.show()

def plot_prediction_difference(y_test, y_pred):
    """Plot the difference between actual and predicted values."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))
    colors = ['blue', 'green', 'orange']

    for i, color in enumerate(colors):
        error = 100*(y_pred[:, i] - y_test[:, i]) / y_test[:, i]
        axs[i].scatter(y_test[:, i], error, color=color, alpha=0.6)
        axs[i].axhline(y=0, color='red', linestyle="--")
        axs[i].set_xlabel(f'Actual Youngs_Modulus{i+1}')
        axs[i].set_ylabel('Prediction Error')
        axs[i].set_ylim((-3, 3))
        axs[i].set_title(f'Youngs_Modulus{i+1}: Prediction Error')

    plt.tight_layout()
    plt.show()

# Load data
# df = pd.read_csv("train_data.csv")  # Replace with actual file path
df = pd.read_csv("2025_3_3_intermediate - Copy(in)_2025_3_5_modified_train.csv")  # Replace with your actual file path

# Define features and target
# X = df[['inner_y1', 'inner_y2', 'inner_y3', 'outer_y1', 'outer_y2', 'outer_y3', 'innerShape_x1', 'innerShape_x2', 'innerShape_x3']].values
X = df[['Principal Component 1 Inner Radius','Principal Component 2 Inner Radius','Principal Component 3 Inner Radius','Principal Component 1 Outer Radius','Principal Component 2 Outer Radius','Principal Component 3 Outer Radius','Principal Component 1 Bottom Cylinder','Principal Component 2 Bottom Cylinder','Principal Component 3 Bottom Cylinder']].values
y = df[['Part1_E', 'Part3_E', 'Part11_E']].values  # Adjust column names as per dataset

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Evaluate
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print(f'Mean Squared Error: {mse:.6f}')

# Plot results
plot_actual_vs_predicted(y_test, y_pred)
plot_prediction_difference(y_test, y_pred)


# Select one feature to vary (e.g., 'inner_y1') and keep others constant
feature_idx = 0  # Index of the feature to vary, 'inner_y1' in this case
fixed_values = np.mean(X_train, axis=0)  # Mean values for all features to keep them constant

# Generate values for the chosen feature ('inner_y1')
x_range = np.linspace(X_train[:, feature_idx].min(), X_train[:, feature_idx].max(), 100)

# Create a new array where only the selected feature varies
X_plot = np.tile(fixed_values, (100, 1))  # Copy fixed values for all features
X_plot[:, feature_idx] = x_range  # Update only the chosen feature

# Apply polynomial feature transformation
X_plot_poly = poly.transform(X_plot)

# Predict Young's modulus values using the trained model
y_plot = model.predict(X_plot_poly)

# Define the range for the plot based on actual values of y_test
y_min, y_max = y_test.min()-0.2, y_test.max()+0.2  # Adding small buffer for better visualization

# Plot the predicted results
fig, axs = plt.subplots(3, 1, figsize=(8, 15))

# Plot for each Young's modulus
for i, (color, title) in enumerate(zip(['blue', 'green', 'orange'], ['Youngs_Modulus1', 'Youngs_Modulus2', 'Youngs_Modulus3'])):
    axs[i].scatter(X_train[:, feature_idx], y_train[:, i], color='gray', alpha=0.5, label="Training Data")
    axs[i].plot(x_range, y_plot[:, i], color=color, linewidth=2, label="Best-Fit Curve")
    axs[i].set_xlabel("inner_y1")  # Label for the varying feature
    axs[i].set_ylabel(title)  # Young's modulus title for the subplot
    axs[i].set_ylim(y_min, y_max)  # Adjust y-limits based on actual values
    axs[i].set_title(f"Polynomial Regression Fit for {title}")
    axs[i].legend()
    axs[i].grid()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
