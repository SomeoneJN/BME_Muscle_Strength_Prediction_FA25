# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:13:40 2025

@author: mgordon
"""


import pandas as pd
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import Plot_intermediate_points as pip
import time
import ShapeAnalysisVerification as sav
import joblib

# pass the headers to be read
# What I think needs to be done long term:
    # Machine Learning Code
    # Load the data
    # separate
    # Train the model
    # test
    # add noise
    # store in correct format
    # test



def read_data_columns(csv_filepath, headers):
    """
    Reads columns from a CSV based on a list of headers, dynamically identifying prefixes.

    Args:
        csv_filepath (str): The path to the CSV file.
        headers (list): A list of column headers or prefixes to read.

    Returns:
        dict: A dictionary of DataFrames/Series, where keys are prefixes, or None if an error occurs.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return None

    prefix_groups = {}  # Dictionary to store columns by prefix

    for header_or_prefix in headers:
        for col in df.columns:
            match = re.match(r"([a-zA-Z_]+)", col)
            if match:
                prefix = match.group(1).lower()
                if header_or_prefix.lower() == prefix:
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(col)

    if not prefix_groups:
        print("Error: No valid headers found.")
        return None

    result = {}
    for prefix, cols in prefix_groups.items():
        result[prefix] = df[cols]

    return result



def add_noise_to_row(row, noise_level=0.1):
    """
    Adds random noise to each value in a pandas Series (row).

    Args:
        row (pandas.Series): The row to add noise to.
        noise_level (float): The relative level of noise to add (e.g., 0.1 for 10% noise).

    Returns:
        pandas.Series: A new Series with added noise.
    """
    if not isinstance(row, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if noise_level < 0:
        raise ValueError("Noise level must be non-negative.")

    noisy_row = row.copy()  # Create a copy to avoid modifying the original

    for index, value in row.items():
        if pd.api.types.is_numeric_dtype(type(value)):  # Only add noise to numeric values
            noise = np.random.normal(0, noise_level) #adjust noise based on value
            noisy_row[index] = value + noise

    return noisy_row



def combine_noisy_rows_to_dataframe(row1, row2, noise_level1=0.1, noise_level2=0.1):
    """
    Combines two noisy rows into a pandas DataFrame with paired y and z coordinates (sequential order).

    Args:
        row1 (pandas.Series): The first row (y-coordinates).
        row2 (pandas.Series): The second row (z-coordinates).
        noise_level1 (float): The noise level for the first row.
        noise_level2 (float): The noise level for the second row.

    Returns:
        pandas.DataFrame: A DataFrame with 'y' and 'z' columns containing the noisy coordinates.
    """
    if not isinstance(row1, pd.Series) or not isinstance(row2, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")

    noisy_row1 = add_noise_to_row(row1, noise_level1)
    noisy_row2 = add_noise_to_row(row2, noise_level2)

    # Convert to lists to ensure sequential order pairing
    y_values = noisy_row1.tolist()
    z_values = noisy_row2.tolist()

    # Determine the shorter length to avoid IndexError
    length = min(len(y_values), len(z_values))

    # Create a DataFrame with 'y' and 'z' columns based on sequential order
    df = pd.DataFrame({
        'y': y_values[:length],
        'z': z_values[:length]
    })

    return df



def combine_noisy_rows_to_list(row1, row2, noise_level1=0.1, noise_level2=0.1):
    """
    Combines two noisy rows into a list of (y, z) tuples.

    Args:
        row1 (pandas.Series): The first row (y-coordinates).
        row2 (pandas.Series): The second row (z-coordinates).
        noise_level1 (float): The noise level for the first row.
        noise_level2 (float): The noise level for the second row.

    Returns:
        list: A list of (y, z) tuples containing the noisy coordinates.
    """
    if not isinstance(row1, pd.Series) or not isinstance(row2, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")

    noisy_row1 = add_noise_to_row(row1, noise_level1)
    noisy_row2 = add_noise_to_row(row2, noise_level2)

    # Convert to lists to ensure sequential order pairing
    y_values = noisy_row1.tolist()
    z_values = noisy_row2.tolist()

    # Determine the shorter length to avoid IndexError
    length = min(len(y_values), len(z_values))

    # Create a list of (y, z) tuples based on sequential order
    combined_list = list(zip(y_values[:length], z_values[:length]))

    return combined_list


def create_coordinate_dict(row1, row2, noise_level1=0.1, noise_level2=0.1, start_key=0):
    """
    Creates a dictionary of {key: [y, z, 4.0]} from noisy data.

    Args:
        row1 (pandas.Series): The first row (y-coordinates).
        row2 (pandas.Series): The second row (z-coordinates).
        noise_level1 (float): The noise level for the first row.
        noise_level2 (float): The noise level for the second row.
        start_key (int): The starting key value for the dictionary.

    Returns:
        dict: A dictionary of {key: [y, z, 4.0]} pairs.
    """
    if not isinstance(row1, pd.Series) or not isinstance(row2, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")

    noisy_row1 = add_noise_to_row(row1, noise_level1)
    noisy_row2 = add_noise_to_row(row2, noise_level2)

    y_values = noisy_row1.tolist()
    z_values = noisy_row2.tolist()

    length = min(len(y_values), len(z_values))

    coordinate_dict = {}
    key = start_key
    for i in range(length):
        coordinate_dict[key] = [y_values[i], z_values[i], 4.0]
        key += 1

    return coordinate_dict

def process_noisy_rows(row1, row2, noise_level1=0.1, noise_level2=0.1):
    """
    Combines two noisy rows into a list of [y, z] lists.

    Args:
        row1 (pandas.Series): The first row (y-coordinates).
        row2 (pandas.Series): The second row (z-coordinates).
        noise_level1 (float): The noise level for the first row.
        noise_level2 (float): The noise level for the second row.

    Returns:
        list: A list of [y, z] lists.
    """
    if not isinstance(row1, pd.Series) or not isinstance(row2, pd.Series):
        raise TypeError("Both inputs must be pandas Series.")

    noisy_row1 = add_noise_to_row(row1, noise_level1)
    noisy_row2 = add_noise_to_row(row2, noise_level2)

    y_values = noisy_row1.tolist()
    z_values = noisy_row2.tolist()

    length = min(len(y_values), len(z_values))

    combined_list = [[y_values[i], z_values[i]] for i in range(length)]

    return combined_list

def convert_format_numpy_array(input_list):
    """
    Converts the input list to a NumPy array of shape (n, 2).

    Args:
        input_list (list): The input list with mixed types.

    Returns:
        numpy.ndarray: A NumPy array of shape (n, 2).
    """
    y_values = []
    z_values = []

    for i, value in enumerate(input_list):
        if i < len(input_list) // 2:  # First half is y values
            if isinstance(value, np.ndarray):
                y_values.append(value.item())
            else:
                y_values.append(value)
        else:  # Second half is z values
            if isinstance(value, np.ndarray):
                z_values.append(value.item())
            else:
                z_values.append(value)

    # Create a NumPy array from the paired y and z values
    result_array = np.array(list(zip(y_values, z_values)))

    return result_array





def add_noise(filepath, headers_to_read, pca_IR, pca_OR, pca_OB, noise_level_1, noise_level_2):

    BYPASS = False    

    results_dict = read_data_columns(filepath, headers_to_read)
    
    # noise = 0.02
    
    # if results_dict:
    #     # for prefix, dataframe in results_dict.items():
    #     #     print(f"Looping through rows of DataFrame with prefix: {prefix}")
    #     for index, row in results_dict.iterrows():
        
    pc_scores_IR_array = []
    pc_scores_OR_array = []
    pc_scores_OB_array = []
    
    if results_dict:
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(results_dict.values(), axis=1)
    
        # print("Looping through all rows:")
        for index, row in combined_df.iterrows():
    
    
            # print(f"  Row index: {index}")
            # print(f"  Row data:\n{row}")
            # Access specific values in the row:
            # example: print(row['inner_y1'])
            
            
            inner_radius_x_df = results_dict['innershape_x']
            # first_row_y = inner_radius_x_df.iloc[0]
            
            inner_radius_y_df = results_dict['innershape_y']
            # first_row_z = inner_radius_y_df.iloc[0]
            
            combined_df_inner_noise = create_coordinate_dict(inner_radius_x_df.iloc[index], inner_radius_y_df.iloc[index], noise_level_1, noise_level_2)
            
            # print("IR_before: ", inner_radius_x_df.iloc[index], inner_radius_y_df.iloc[index])
            
            # print("IR_after: ", combined_df_inner_noise)
            
            outer_radius_x_df = results_dict['outershape_x']
            # first_row_y = inner_y_df.iloc[0]
            
            outer_radius_y_df = results_dict['outershape_y']
            # first_row_z = inner_z_df.iloc[0]
            
            combined_df_outer_noise = create_coordinate_dict(outer_radius_x_df.iloc[index], outer_radius_y_df.iloc[index], noise_level_1, noise_level_2)
            
            
            spline_points_inner, spline_points_outer = pip.angle_spline_driver(combined_df_inner_noise, combined_df_outer_noise)
            
            
            
            #############################################################
            if BYPASS:
                inner_radius = sav.get_2d_coords_from_dictionary(combined_df_inner_noise)
                # print("inner_radius: ", inner_radius)
                
                inner_radius = np.array(inner_radius)
                spline_points_inner = inner_radius
                
            #############################################################    
            
            
            # print("inner_radius: ", inner_radius)
            
            # print("Spline Points: ", spline_points_inner)
    
            # Calculate the PC Scores for this run for the inner radius
            # Initialize an empty list to store the elements
            # data_row = []
            
            # Iterate through each row (pair) in the original array
            # for row in spline_points_inner:
            #     # Append the first element of the pair
            #     data_row.append(row[0])
            #     # Append the second element of the pair
            #     data_row.append(row[1])
                
                
            first_elements = []
            second_elements = []
            
            for row in spline_points_inner:
                first_elements.append(row[0])
                second_elements.append(row[1])
        
            data_row = first_elements + second_elements
            
            df = pd.DataFrame([data_row])
            
            # print("******df:", df)
            
            pc_scores_IR = pca_IR.transform(df)
    
            # Calculate the PC Scores for this run for the outer radius
            # data_row = []
            
            # # Iterate through each row (pair) in the original array
            # for row in spline_points_outer:
            #     # Append the first element of the pair
            #     data_row.append(row[0])
            #     # Append the second element of the pair
            #     data_row.append(row[1])
            
            first_elements = []
            second_elements = []
            
            for row in spline_points_outer:
                first_elements.append(row[0])
                second_elements.append(row[1])
        
            data_row = first_elements + second_elements
            
            df = pd.DataFrame([data_row])
            
            pc_scores_OR = pca_OR.transform(df)
            
            
            outer_y_df = results_dict['inner_y']
            # outer_y_df = results_dict['outer_y']
            # first_row_y = outer_y_df.iloc[0]
            
            # print(outer_y_df)
            
            outer_z_df = results_dict['inner_z']
            # outer_z_df = results_dict['outer_z']
            # first_row_z = outer_z_df.iloc[0]
    
            # combined_df_outer = create_coordinate_dict(outer_y_df.iloc[index], outer_z_df.iloc[index], noise_level1=noise, noise_level2=noise)
            # combined_df_outer_NN = create_coordinate_dict(first_row_y, first_row_z, noise_level1=noise, noise_level2=noise)
            
            # print("outer_y_df", outer_y_df)
            combined_list_outer = process_noisy_rows(outer_y_df.iloc[index], outer_z_df.iloc[index], noise_level_1, noise_level_2)
            # print("combined_list_outer", combined_list_outer)
            
            outer_pc_points = sav.generate_2d_coords_for_cylinder_pca(combined_list_outer, 9) #REPLACE WITH CYLINDER POINTS
            
            # print("outer_pc_points:", outer_pc_points)
            
            outer_pc_points_converted = convert_format_numpy_array(outer_pc_points)
            
            
            # data_row = []
            
            # # Iterate through each row (pair) in the original array
            # for row in outer_pc_points_converted:
            #     # Append the first element of the pair
            #     data_row.append(row[0])
            #     # Append the second element of the pair
            #     data_row.append(row[1])
            
            first_elements = []
            second_elements = []
            
            for row in outer_pc_points_converted:
                first_elements.append(row[0])
                second_elements.append(row[1])
        
            data_row = first_elements + second_elements
            
            df = pd.DataFrame([data_row])

            # print("outer bottom df", df)

            #######################################
            if BYPASS:
                # print(outer_y_df.iloc[index])
                # df =  pd.concat([outer_y_df.iloc[index], outer_z_df.iloc[index]], axis=0)   
# result = pd.concat([df1, df2], axis=1)                 
                
                y_df = outer_y_df.iloc[index]
                z_df = outer_z_df.iloc[index]
                
                # print(y_df.shape)
                # print(z_df.shape)

                
                values1 = y_df.values
                values2 = z_df.values
            
                combined_values = np.concatenate((values1, values2))
            
                new_df = pd.DataFrame([combined_values])
                new_df.index = ['outer bottom df']
                new_df.columns = np.arange(len(combined_values))
                
                df = new_df
                # print("bypassed df:", df)
            #######################################
            
            
            pc_scores_OB = pca_OB.transform(df)
            
            pc_scores_IR_array.append(pc_scores_IR)
            pc_scores_OR_array.append(pc_scores_OR)
            pc_scores_OB_array.append(pc_scores_OB)
            
            
            
            # X_train_pca = np.hstack([
            #     pc_scores_IR,
            #     pc_scores_OR,
            #     pc_scores_OB
            # ])
            
            # print(X_train_pca)
            
            # print(inner_pc_points_converted)
            # print(type(inner_pc_points_converted))
            
            # plt.figure(figsize=(8, 6))
            # plt.scatter(combined_df_inner[:,0], combined_df_inner[:,1])
            # # plt.scatter(first_row_y, first_row_z)
            # # plt.xlabel("Noisy Y Values")
            # # plt.ylabel("Noisy Z Values")
            # # plt.title("Noisy Y vs. Noisy Z")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
                
            
    else:
        print("Error: results_dict is empty or None")
    
    
    pc_scores_IR_array = np.concatenate(pc_scores_IR_array, axis = 0)
    pc_scores_OR_array = np.concatenate(pc_scores_OR_array, axis = 0)
    pc_scores_OB_array = np.concatenate(pc_scores_OB_array, axis = 0)
    
    return(pc_scores_IR_array, pc_scores_OR_array, pc_scores_OB_array)


# # # Example usage:
# # filepath = r"C:\Users\mgordon\My Drive\a  Research\a Pelvic Floor\Inverse FEA\SU24 Analysis Code\2025_3_3_intermediate - Copy(in)_2025_3_5_modified_train.csv"
# # filepath = r"C:\Users\mgordon\My Drive\a  Research\a Pelvic Floor\Inverse FEA\SU24 Analysis Code\Test.csv"

# filepath = r"C:\Users\mgordon\My Drive\a  Research\a Pelvic Floor\Inverse FEA\SU24 Analysis Code\test_data.csv"

# pca_IR = joblib.load('pcaIR.joblib')
# pca_OR = joblib.load('pcaOR.joblib')
# pca_OB = joblib.load('pcaOB.joblib')

# noise_level_1 = 0
# noise_level_2 = noise_level_1


# headers_to_read = ["innerShape_x", "innerShape_y", "outerShape_x", "outerShape_y", "inner_y", "inner_z", "outer_y", "outer_z"] #example headers

# pc_scores_IR_array, pc_scores_OR_array, pc_scores_OB_array = add_noise(filepath, headers_to_read, pca_IR, pca_OR, pca_OB, noise_level_1, noise_level_2)