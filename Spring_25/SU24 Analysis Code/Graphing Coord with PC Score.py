# import pandas as pd
# import re
# import matplotlib.pyplot as plt

# def read_data(csv_filepath):
#     """Reads inner/outer y/z and PC1 columns from a CSV."""
#     try:
#         df = pd.read_csv(csv_filepath)
#     except FileNotFoundError:
#         print(f"Error: File not found at {csv_filepath}")
#         return None, None, None, None, None

#     inner_y_cols = [col for col in df.columns if re.match(r"inner_y.*", col, re.IGNORECASE)]
#     inner_z_cols = [col for col in df.columns if re.match(r"inner_z.*", col, re.IGNORECASE)]
#     outer_y_cols = [col for col in df.columns if re.match(r"outer_y.*", col, re.IGNORECASE)]
#     outer_z_cols = [col for col in df.columns if re.match(r"outer_z.*", col, re.IGNORECASE)]

#     pc1_column = None
#     for col in df.columns:
#         if "principal component 1" in col.lower():  # Match the exact name (case-insensitive)
#             pc1_column = col
#             break

#     if not inner_y_cols and not inner_z_cols and not outer_y_cols and not outer_z_cols and pc1_column is None:
#         print("Error: No columns found matching 'inner/outer y/z' or 'principal component 1'.")
#         return None, None, None, None, None

#     y_df = df[inner_y_cols] if inner_y_cols else pd.DataFrame()
#     z_df = df[inner_z_cols] if inner_z_cols else pd.DataFrame()
#     outer_y_df = df[outer_y_cols] if outer_y_cols else pd.DataFrame()
#     outer_z_df = df[outer_z_cols] if outer_z_cols else pd.DataFrame()
#     pc1_data = df[pc1_column] if pc1_column is not None else pd.Series()

#     return y_df, z_df, outer_y_df, outer_z_df, pc1_data


# # Example usage:
# filepath = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing\2025_2_21_intermediate_2025_2_21_modified_train.csv"
# y_data, z_data, outer_y_data, outer_z_data, pc1_values = read_data(filepath)

# if all(data is not None for data in [y_data, z_data, outer_y_data, outer_z_data, pc1_values]):
#     if not y_data.empty and not z_data.empty and len(y_data) == len(z_data) == len(pc1_values) and not pc1_values.empty and len(y_data.columns) > 0 and len(z_data.columns) > 0:
#         inner_y = y_data.iloc[:, 0]  # First inner_y column
#         inner_z = z_data.iloc[:, 0]  # First inner_z column

#         plt.figure(figsize=(8, 6))
#         plt.scatter(inner_y, inner_z, c=pc1_values, cmap='viridis', s=20)
#         plt.xlabel("Inner Y")
#         plt.ylabel("Inner Z")
#         plt.title("Inner Y vs. Inner Z (Colored by Principal Component 1)")
#         plt.colorbar(label="Principal Component 1 Values")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     else:
#         print("Error: Inner Y/Z data missing, lengths differ, or Principal Component 1 values missing or empty columns.")

# elif any(data is None for data in [y_data, z_data, outer_y_data, outer_z_data, pc1_values]):
#     print("An error occurred during file reading or some columns were not found.")


import pandas as pd
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Set the interactive backend (before creating any figures)
# plt.switch_backend('qt5agg')  # Or 'TkAgg', 'wxAgg' - experiment to see which works best for you.




def read_data(csv_filepath):
    """Reads inner/outer y/z and PC1 columns from a CSV."""
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return None, None, None, None, None

    # inner_y_cols = [col for col in df.columns if re.match(r"inner_y.*", col, re.IGNORECASE)]
    # inner_z_cols = [col for col in df.columns if re.match(r"inner_z.*", col, re.IGNORECASE)]
    # outer_y_cols = [col for col in df.columns if re.match(r"outer_y.*", col, re.IGNORECASE)]
    # outer_z_cols = [col for col in df.columns if re.match(r"outer_z.*", col, re.IGNORECASE)]

    inner_y_cols = [col for col in df.columns if re.match(r"innerShape_x.*", col, re.IGNORECASE)]
    inner_z_cols = [col for col in df.columns if re.match(r"innerShape_y.*", col, re.IGNORECASE)]
    outer_y_cols = [col for col in df.columns if re.match(r"outerShape_x.*", col, re.IGNORECASE)]
    outer_z_cols = [col for col in df.columns if re.match(r"outerShape_y.*", col, re.IGNORECASE)]

    pc1_column = None
    for col in df.columns:
        if "principal component 1 inner radius" in col.lower():  # Match the exact name (case-insensitive)
            pc1_column = col
            break

    print(inner_y_cols)
    print(inner_z_cols)
    print(outer_y_cols)
    print(outer_z_cols)
    print(pc1_column)
    
    if not inner_y_cols and not inner_z_cols and not outer_y_cols and not outer_z_cols and pc1_column is None:
        print("Error: No columns found matching 'inner/outer y/z' or 'principal component 1'.")
        return None, None, None, None, None

    y_df = df[inner_y_cols] if inner_y_cols else pd.DataFrame()
    z_df = df[inner_z_cols] if inner_z_cols else pd.DataFrame()
    outer_y_df = df[outer_y_cols] if outer_y_cols else pd.DataFrame()
    outer_z_df = df[outer_z_cols] if outer_z_cols else pd.DataFrame()
    pc1_data = df[pc1_column] if pc1_column is not None else pd.Series()

    return y_df, z_df, outer_y_df, outer_z_df, pc1_data


# Example usage:
filepath = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing\2025_2_22_intermediate_2025_2_22_modified_train.csv"
y_data, z_data, outer_y_data, outer_z_data, pc1_values = read_data(filepath)


# ########################### 3D Graph
# if all(data is not None for data in [y_data, z_data, outer_y_data, outer_z_data, pc1_values]):
#     if not y_data.empty and not z_data.empty and len(y_data.columns) > 0 and len(z_data.columns) > 0 and len(y_data.columns) == len(z_data.columns) and not pc1_values.empty and len(y_data) == len(pc1_values):
#         inner_y_row = y_data.iloc[0, :]
#         inner_z_row = z_data.iloc[0, :]

#         min_pc1 = pc1_values.min()
#         max_pc1 = pc1_values.max()
    
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
    
#         norm = plt.Normalize(vmin=min_pc1, vmax=max_pc1)
#         scalarMap = cm.ScalarMappable(norm=norm, cmap='viridis')
#         scalarMap.set_array([])
    
#         for i in range(len(y_data)):
#             inner_y_row = y_data.iloc[i, :]
#             inner_z_row = z_data.iloc[i, :]
#             pc1_value = pc1_values.iloc[i]  # Correct: Inside the loop!
    
#             scatter = ax.scatter(inner_y_row, inner_z_row, [pc1_value] * len(inner_y_row), c=[pc1_value] * len(inner_y_row), cmap='viridis', s=20, label=f"Row {i+1}") # Assign to scatter

    
#         ax.set_xlabel("Inner Y Value")
#         ax.set_ylabel("Inner Z Value")
#         ax.set_zlabel("Principal Component 1 Value")
#         ax.set_title("Inner Y vs. Inner Z vs. Principal Component 1 (All Rows)")
    
#      # Create the colorbar using the last scatter plot
#         cbar = fig.colorbar(scatter, label="Principal Component 1 Value") # Use the scatter plot from the last iteration

    
#         ax.grid(True)
#         ax.legend()
#         plt.tight_layout()
#         plt.show()

#     else:
#         print("Error: Inner Y/Z data missing or empty dataframes or columns, or columns don't match, or pc1_values is empty.")



# elif any(data is None for data in [y_data, z_data, outer_y_data, outer_z_data, pc1_values]):
#     print("An error occurred during file reading or some columns were not found.")
    
    
    
    
#################### 2D Graph #############################

if all(data is not None for data in [y_data, z_data, outer_y_data, outer_z_data, pc1_values]):
    # Define min_pc1 and max_pc1 *before* the if statement
    if not pc1_values.empty:
        min_pc1 = pc1_values.min()
        max_pc1 = pc1_values.max()
    else: #handle if pc1_values is empty
        min_pc1 = 0
        max_pc1 = 1

    if not y_data.empty and not z_data.empty and len(y_data.columns) > 0 and len(z_data.columns) > 0 and len(y_data.columns) == len(z_data.columns) and not pc1_values.empty and len(y_data) == len(pc1_values):
        # ... (rest of the plotting code - same as before)
        fig, ax = plt.subplots(figsize=(10, 8))

        norm = plt.Normalize(vmin=min_pc1, vmax=max_pc1)

        for i in range(len(y_data)):  # Iterate through all rows
            inner_y_row = y_data.iloc[i,:]
            inner_z_row = z_data.iloc[i,:]
            pc1_value = pc1_values.iloc[i]

            scatter = ax.scatter(inner_y_row, inner_z_row, c=[pc1_value] * len(inner_y_row), cmap='viridis', s=20, norm=norm, label=f"Row {i+1}")

        ax.set_xlabel("Inner Y Value")
        ax.set_ylabel("Inner Z Value")
        ax.set_title("Inner Y vs. Inner Z (All Rows)")

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Principal Component 1 Value")

        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    else:
        print("Error: Inner Y/Z data missing or empty dataframes or columns, or columns/lengths don't match, or pc1_values is empty.")


elif any(data is None for data in [y_data, z_data, outer_y_data, outer_z_data, pc1_values]):
    print("An error occurred during file reading or some columns were not found.")
