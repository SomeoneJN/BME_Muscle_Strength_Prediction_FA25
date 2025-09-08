"""
This file is designed to use the IO functions relating to FeBio, which can entail feb files that are XML input files,
and log files that are txt output files. The goal is to replicate the same process as the post-processing currently
implemented for the INP files with Abaqus.
"""
import PCA_data
import generate_pca_points_AVW as gic
import time
import csv
import pandas as pd
import os
import predict_functions as pf
import ShapeAnalysisVerification as sav
import numpy as np
import CylinderFunctions

# TODO: Change the following to preferred numbers
window_width = 0.3
num_pts = 9
spline_ordered = 0
startingPointColHeader = 'inner_y'
secondPartStart = 'outer_x'
numCompPCA = 3

# TODO: Replace Headers when changing what intermediate displays
# intermediate_Headers = ['inner_y','inner_z', 'outer_y', 'outer_z', 'innerShape_x', 'innerShape_y', 'outerShape_x', 'outerShape_y']

intermediate_Headers = ['inner_base_y','inner_base_z', 'outer_base_y', 'outer_base_z', 'inner_circle_x', 'inner_circle_y', 'outer_circle_x', 'outer_circle_y']
PCA_Headers = ['inner_y', 'outer_y', 'innerShape_x', 'outerShape_x']
Modified_Train_Headers = ["File Name", "Part1_E", "Part3_E", "Part11_E", "Pressure", "Inner_Radius", "Outer_Radius"]





import pandas as pd

# Modified section of PostProcess_FeBio.py

def process_features(csv_file, Results_Folder, date_prefix, numCompPCA):
    """
    Reads data from a CSV file and groups the columns into three blocks:
      Block 1: Bottom Cylinder – consists of inner_y and inner_z columns.
      Block 2: Inner Shape    – consists of innerShape_x and innerShape_y columns.
      Block 3: Outer Shape    – consists of outerShape_x and outerShape_y columns.
    
    PCA is then applied independently to each block (retaining numCompPCA components),
    and the resulting PCA scores are appended to the original data.
    
    The combined DataFrame is saved as a modified CSV file.
    
    Returns:
       file_path: Path to the saved modified CSV.
       pca_inner: PCA model for the inner shape.
       pca_outer: PCA model for the outer shape.
       pca_bottom: PCA model for the bottom cylinder.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Define groups along with a matching function for each:
    groups = [
       ("bottom", lambda col: col.startswith("inner_y") or col.startswith("inner_z")),
       ("innerShape", lambda col: col.startswith("innerShape_")),
       ("outerShape", lambda col: col.startswith("outerShape_"))
    ]
    
    group_dfs = {}
    for group_name, match_fn in groups:
        matching_cols = [col for col in df.columns if match_fn(col)]
        if not matching_cols:
            print(f"Warning: No columns found for group '{group_name}'")
            continue
        # Get the indices in order of appearance and extract the contiguous block.
        indices = sorted([df.columns.get_loc(col) for col in matching_cols])
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        group_dfs[group_name] = df.iloc[:, start_idx:end_idx]
        #print(f"Extracted group '{group_name}' with columns: {df.columns[start_idx:end_idx].tolist()}")

    # Ensure that all required groups are present.
    for req in ["bottom", "innerShape", "outerShape"]:
        if req not in group_dfs:
            print(f"Error: Required group '{req}' is missing for PCA.")
            return None

    # Apply PCA to each group using your PCA_data.PCA_ function.
    total_result_bottom, pca_bottom = PCA_data.PCA_(group_dfs["bottom"], numCompPCA)
    total_result_innerShape, pca_inner = PCA_data.PCA_(group_dfs["innerShape"], numCompPCA)
    total_result_outerShape, pca_outer = PCA_data.PCA_(group_dfs["outerShape"], numCompPCA)

    # Get the PCA score columns from each block.
    bottom_score_cols     = [f"PC{i+1}_Bottom" for i in range(numCompPCA)]
    innerShape_score_cols = [f"PC{i+1}_InnerShape" for i in range(numCompPCA)]
    outerShape_score_cols = [f"PC{i+1}_OuterShape" for i in range(numCompPCA)]

    bottom_scores     = total_result_bottom.iloc[:, -numCompPCA:]
    innerShape_scores = total_result_innerShape.iloc[:, -numCompPCA:]
    outerShape_scores = total_result_outerShape.iloc[:, -numCompPCA:]
    
    bottom_scores.columns     = bottom_score_cols
    innerShape_scores.columns = innerShape_score_cols
    outerShape_scores.columns = outerShape_score_cols

    # Append the PCA score columns to the original DataFrame.
    final_df = pd.concat([df, bottom_scores, innerShape_scores, outerShape_scores], axis=1)

    if not os.path.exists(Results_Folder):
         os.makedirs(Results_Folder)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    file_path = os.path.join(Results_Folder, f'{base_name}_{date_prefix}_modified_train.csv')
    final_df.to_csv(file_path, index=False)
    
    return file_path, pca_inner, pca_outer, pca_bottom


# def process_features(csv_file, Results_Folder, date_prefix, numCompPCA):
#     """
#     Reads data from a CSV file, extracting specific columns based on PCA headers 
#     and storing each section in its own DataFrame with specific names.

#     Args:
#         csv_file (str): The path to the CSV file.

#     Returns:
#         tuple: A tuple containing the four DataFrames (pc_inner_bottom_df, 
#                pc_outer_bottom_df, pc_inner_radius_df, pc_outer_radius_df),
#                or None if an error occurs.
#     """

#     PCA_Headers = ['inner_y', 'outer_y', 'innerShape_x', 'outerShape_x']
#     # df_names = ['pc_inner_radius_df', 'pc_outer_radius_df', 'pc_inner_bottom_df', 'pc_outer_bottom_df'] # Corresponding names

#     # dfs = {}

#     try:
#         df = pd.read_csv(csv_file)
#     except FileNotFoundError:
#         print(f"Error: CSV file not found at {csv_file}")
#         return None
#     except Exception as e:
#         print(f"Error reading CSV: {e}")
#         return None

#     for i, header in enumerate(PCA_Headers):
#         matching_cols = [col for col in df.columns if col.startswith(header)]
#         print("header: ", header)
#         print(matching_cols)
#         if not matching_cols:
#             print(f"Warning: No columns found starting with '{header}'")
#             continue

#         start_index = df.columns.get_loc(matching_cols[0])
#         print(start_index)

#         # Find the *next* relevant header, even if it's not the next in PCA_Headers
#         next_header_index = len(df.columns)  # Initialize to end of DataFrame

#         for next_h in PCA_Headers:
#             if next_h != header:
#                 next_matching_cols = [col for col in df.columns if col.startswith(next_h)]
#                 if next_matching_cols:
#                     try:
#                         next_h_index = df.columns.get_loc(next_matching_cols[0])
#                         if next_h_index > start_index:  # Crucial check!
#                             next_header_index = min(next_header_index, next_h_index)

#                     except KeyError:
#                         pass # next header might not be in file

        
#         extracted_block = df.iloc[:, start_index:next_header_index]
#         print("extracted_block: ", extracted_block)
#         # df_names[i] = extracted_block
#         if i == 0:
#             pc_inner_radius_df = extracted_block
#         elif i == 1:
#             pc_outer_radius_df = extracted_block
#         elif i == 2:
#             pc_inner_inner_bottom_df= extracted_block
#         else:
#             pc_inner_outer_bottom_df = extracted_block
#         # dfs[df_names[i]] = extracted_block

#     # if not dfs:
#     #     print("No PCA data found in the CSV.")
#     #     return None

#     # print("*********************************")
#     # print(dfs.get('pc_inner_radius_df'))
#     # print("*********************************")
#     # print(dfs.get('pc_outer_radius_df'))
#     # print("*********************************")
#     # print(dfs.get('pc_inner_bottom_df'))
#     # print("*********************************")
#     # print(dfs.get('pc_outer_bottom_df'))
    
#     print("*********************************")
#     print('pc_inner_radius_df')
#     print("*********************************")
#     print('pc_outer_radius_df')
#     print("*********************************")
#     print('pc_inner_bottom_df')
#     print("*********************************")
#     print('pc_outer_bottom_df')

#     # # Return the DataFrames as a tuple
#     # return (dfs.get('pc_inner_radius_df'), dfs.get('pc_outer_radius_df'), 
#     #         dfs.get('pc_inner_bottom_df'), dfs.get('pc_outer_bottom_df'))

#     # # Perform PCA on the sliced DataFrames
#     # total_result_PCIR, pcaIR = PCA_data.PCA_(dfs.get('pc_inner_radius_df'), numCompPCA)
#     # total_result_PCOR, pcaOR = PCA_data.PCA_(dfs.get('pc_outer_radius_df'), numCompPCA)
#     # total_result_PCOB, pcaOB = PCA_data.PCA_(dfs.get('pc_outer_bottom_df'), numCompPCA)
    
#     # Perform PCA on the sliced DataFrames
#     total_result_PCIR, pcaIR = PCA_data.PCA_(pc_inner_radius_df, numCompPCA)
#     total_result_PCOR, pcaOR = PCA_data.PCA_(pc_outer_radius_df, numCompPCA)
#     total_result_PCOB, pcaOB = PCA_data.PCA_(pc_inner_outer_bottom_df, numCompPCA)
#     # print('*******************', pc_inner_radius_df)


#     # Previous code that doesn't seem to be getting the actual PCs
#     # # Get the principal component scores
#     # PC_scores = total_result_PC1.iloc[:, :numCompPCA]
#     # PC_scores_bottom = total_result_PCB.iloc[:, :numCompPCA]
#     # PC_scores_radius = total_result_PCR.iloc[:, :numCompPCA]



#     # Get the principal component scores
#     PC_scores_IR = total_result_PCIR.iloc[:, -1*numCompPCA:]
#     PC_scores_OR = total_result_PCOR.iloc[:, -1*numCompPCA:]
#     PC_scores_OB = total_result_PCOB.iloc[:, -1*numCompPCA:]



#     # print("total_result_PC1: ", total_result_PC1)
#     # # print("PC_scores: ", PC_scores)
#     # print("numCompPCA", numCompPCA)

#     # Rename the column headers to "Principal Component i Inner/Outer Radius"
#     # PC_scores_IR = PC_scores_IR.rename(
#     #     columns={f'inner_x{i + 1}': f'Principal Component {i + 1} Inner Radius' for i in range(numCompPCA)})

#     # PC_scores_OR = PC_scores_OR.rename(
#     #     columns={f'outer_x{i + 1}': f'Principal Component {i + 1} Outer Radius' for i in range(numCompPCA)})

#     # PC_scores_OB = PC_scores_OB.rename(
#     #     columns={f'radius{i + 1}' : f'Principal Component {i + 1} Cylinder' for i in range(numCompPCA)})
    
    
#     PC_scores_IR = PC_scores_IR.rename(
#         columns={f'principal component {i + 1}': f'Principal Component {i + 1} Inner Radius' for i in range(numCompPCA)})

#     PC_scores_OR = PC_scores_OR.rename(
#         columns={f'principal component {i + 1}': f'Principal Component {i + 1} Outer Radius' for i in range(numCompPCA)})

#     PC_scores_OB = PC_scores_OB.rename(
#         columns={f'principal component {i + 1}' : f'Principal Component {i + 1} Bottom Cylinder' for i in range(numCompPCA)})
    
#     print("**************************")
#     print(PC_scores_IR)

#     # Concatenate the DataFrames to create the final DataFrame
#     # final_df = pd.concat([df.loc[:, Modified_Train_Headers],
#     #                       PC_scores_IR, PC_scores_OR, PC_scores_OB], axis=1)

#     final_df = pd.concat([df.loc[:, :],
#                           PC_scores_IR, PC_scores_OR, PC_scores_OB], axis=1)


#     # print("final_df: ", final_df)

#     # Create the directory if it doesn't exist
#     if not os.path.exists(Results_Folder):
#         os.makedirs(Results_Folder)

#     # Get the base file name
#     file_name = pf.get_file_name(csv_file)

#     # Construct the file path for the modified CSV
#     file_path = os.path.join(Results_Folder, f'{file_name}_{date_prefix}_modified_train.csv')

#     # Save the final DataFrame to a CSV file
#     final_df.to_csv(file_path, index=False)

#     # Return the file path and PCA models
#     return file_path, pcaIR, pcaOR, pcaOB


# # Example usage:
# csv_file_path = 'your_file.csv'  # Replace with your CSV file path
# (pc_inner_radius_df, pc_outer_radius_df, pc_inner_bottom_df, pc_outer_bottom_df) = process_features_into_named_dfs(csv_file_path)

# if pc_inner_radius_df is not None:
#     print("pc_inner_radius_df:")
#     print(pc_inner_radius_df.head())

# if pc_outer_radius_df is not None:
#     print("\npc_outer_radius_df:")
#     print(pc_outer_radius_df.head())

# if pc_inner_bottom_df is not None:
#     print("\npc_inner_bottom_df:")
#     print(pc_inner_bottom_df.head())

# if pc_outer_bottom_df is not None:
#     print("\npc_outer_bottom_df:")
#     print(pc_outer_bottom_df.head())


# # Example of how to check if a DataFrame exists and use it
# if pc_inner_radius_df is not None:
#     # ... process pc_inner_radius_df ...
#     pass

# if pc_inner_bottom_df is not None:
#   # ... process pc_inner_bottom_df
#   pass

# return pc_outer_bottom_df
# ... etc. for the other DataFrames



"""
    Generate modified training CSV files with principal component scores from the original file.

    Parameters:
        csv_file (str): The path to the original CSV file.
        Results_Folder (str): The folder path where the modified CSV file will be saved.
        date_prefix (str): The prefix to be added to the modified CSV file name.
        numCompPCA (int): The number of principal components to retain.

    Returns:
        file_path (str): The path of the generated modified training CSV file.
        pca1 (object): The PCA model object for the top tissue.
        pcaB (object): The PCA model object for the bottom tissue.

    Example:
        >>> csv_file = "data.csv"
        >>> Results_Folder = "results"
        >>> date_prefix = "2024_05_08"
        >>> numCompPCA = 3
        >>> file_path, pca1, pcaB = process_features(csv_file, Results_Folder, date_prefix, numCompPCA)
    """
def process_features_previous(csv_file, Results_Folder, date_prefix, numCompPCA):
    
    print("csv_file",csv_file)
    print("Results_Folder",Results_Folder)
    print("date_prefix",date_prefix)
    print("numCompPCA",numCompPCA)
    
    # Read the input CSV file into a pandas DataFrame
    int_df = pd.read_csv(csv_file)

    # Iterate through the headers
    print("PCA_Headers", PCA_Headers)
    for i, header in enumerate(PCA_Headers):
        # If not the last header, get the next header
        if i < len(PCA_Headers) - 1:
            next_header = PCA_Headers[i + 1]
            final_header = PCA_Headers[i + 2]
            print("This is next header: ", next_header)
            print("This is final header: ", next_header)

        # Get the start and end indices of the current header's columns
        currentStartIndex = int_df.columns[int_df.columns.str.contains(header)].tolist()
        print("currentStartIndex",currentStartIndex)
        currentIndex = int_df.columns.get_loc(currentStartIndex[0])


        nextStartIndex = int_df.columns[int_df.columns.str.contains(next_header)].tolist()
        nextIndex = int_df.columns.get_loc(nextStartIndex[0])


        final_Start_Index = int_df.columns[int_df.columns.str.contains(final_header)].tolist()
        final_Index = int_df.columns.get_loc(final_Start_Index[0])

        # Slice the DataFrame to get the columns for the current header
        pc1_df = int_df.iloc[:, currentIndex:nextIndex]  # TODO: HARD CODED _ CHANGE LATER
        pc_outer_bottom = int_df.iloc[:, nextIndex:final_Index]
        pc_radius_df = int_df.iloc[:, final_Index:len(int_df.columns)]

        # Perform PCA on the sliced DataFrames
        total_result_PC1, pca1 = PCA_data.PCA_(pc1_df, numCompPCA)
        total_result_PCB, pcaB = PCA_data.PCA_([pc_outer_bottom], numCompPCA)
        total_result_PCR, pcaR = PCA_data.PCA_([pc_radius_df], numCompPCA)


        # Previous code that doesn't seem to be getting the actual PCs
        # # Get the principal component scores
        # PC_scores = total_result_PC1.iloc[:, :numCompPCA]
        # PC_scores_bottom = total_result_PCB.iloc[:, :numCompPCA]
        # PC_scores_radius = total_result_PCR.iloc[:, :numCompPCA]

        # Get the principal component scores
        PC_scores = total_result_PC1.iloc[:, -1*numCompPCA:]
        PC_scores_bottom = total_result_PCB.iloc[:, -1*numCompPCA:]
        PC_scores_radius = total_result_PCR.iloc[:, -1*numCompPCA:]



        print("total_result_PC1: ", total_result_PC1)
        # print("PC_scores: ", PC_scores)
        print("numCompPCA", numCompPCA)

        # Rename the column headers to "Principal Component i Inner/Outer Radius"
        PC_scores = PC_scores.rename(
            columns={f'inner_x{i + 1}': f'Principal Component {i + 1} Inner Radius' for i in range(numCompPCA)})

        PC_scores_bottom = PC_scores_bottom.rename(
            columns={f'outer_x{i + 1}': f'Principal Component {i + 1} Outer Radius' for i in range(numCompPCA)})

        PC_scores_radius = PC_scores_radius.rename(
            columns={f'radius{i + 1}' : f'Principal Component {i + 1} Cylinder' for i in range(numCompPCA)})
        
        # print(PC_scores)

        # Concatenate the DataFrames to create the final DataFrame
        final_df = pd.concat([int_df.loc[:, Modified_Train_Headers],
                              PC_scores, PC_scores_bottom, PC_scores_radius], axis=1)

        # final_df = pd.concat([int_df.loc[:, :],
        #                       PC_scores, PC_scores_bottom, PC_scores_radius], axis=1)

        # print("final_df: ", final_df)

        # Create the directory if it doesn't exist
        if not os.path.exists(Results_Folder):
            os.makedirs(Results_Folder)

        # Get the base file name
        file_name = pf.get_file_name(csv_file)

        # Construct the file path for the modified CSV
        file_path = os.path.join(Results_Folder, f'{file_name}_{date_prefix}_modified_train.csv')

        # Save the final DataFrame to a CSV file
        final_df.to_csv(file_path, index=False)

        # Return the file path and PCA models
        return file_path, pca1, pcaB, pcaR


def find_apex(coordList):
    min_y = coordList[0][1][1]
    for coord in coordList:
        if coord[1][1] < min_y:
            min_y = coord[1][1]

    return min_y




"""
This function was originally made to generate a csv intermediate file for FeBio post-processing. Its original purpose
was to work with the model from summer of 2023 which had an APEX and pc_points_bottom.

For the summer of 2024 we changed it to work with a cylinder that has Pressure, Inner Radius, Outer Radius. 
this was done by commenting out the line "pc_points_bottom = bts.generate_2d_bottom_tissue(bts.extract_ten_coordinates_block(obj_coords_list[2]))"
and line "apex = find_apex(obj_coords_list[1])" To revert back to old model simply change headers and then uncomment the lines metioned 
above. Also uncomment the second for loop starting at "coord = 'Bx'"
"""
def generate_int_csvs(file_params, object_list, log_name, feb_name, first_int_file_flag, csv_filename, inner_radius_spline, outer_radius_spline, current_run_dict, plot_points_on_spline):
    obj_coords_list = []

    csv_row = []
    csv_header = []

    # Get the pure file name that just has the material parameters
    prop_final = []
    # Get the changed material properties
    for key, value in current_run_dict.items():
        prop = float(value)
        prop_final.append(prop)

    # Get the final coordinates for each object in list from the log file
    
    for obj in object_list:
        print('Extracting... ' + obj + ' for ' + file_params)
        obj_coords_list.append(gic.extract_coordinates_from_final_step(log_name, feb_name, obj))

    #TODO: THIS IS WHERE THE INT CSV DATA IS OBTAINED


    # gets obj_coords_list to [[x, y, z]]
    object_coords_list = []
    for ele in obj_coords_list[0]:
        temparray = []
        temparray.append(ele[1][0])
        temparray.append(ele[1][1])
        temparray.append(ele[1][2])
        object_coords_list.append(temparray)

    # THIS USES THE FUNCTION THAT WE CREATED TO GENERATE THE OUTER AND INNER POINTS THEN PASSING THEM
    # IN TO FIND THE 2D_COORDS FOR THE PCA POINTS
    # print("num_pts", num_pts)
    # print("object_coords_list", object_coords_list)
    # print("window_width", window_width)
    
    outer_points = sav.generate_outer_cylinder_bottom(num_pts, object_coords_list, window_width)

    inner_points = sav.generate_inner_cylinder_bottom(num_pts, object_coords_list, window_width)

    if plot_points_on_spline:

        # TODO: comment out above if do not want those plot.
        inner_radius_pc_points_plot = CylinderFunctions.pair_points(inner_radius_spline)
        outer_radius_pc_points_plot = CylinderFunctions.pair_points(outer_radius_spline)
        CylinderFunctions.plot_pc_points(inner_radius_pc_points_plot)
        CylinderFunctions.plot_pc_points(outer_radius_pc_points_plot)


    print("outer_points", outer_points)
    outer_pc_points = sav.generate_2d_coords_for_cylinder_pca(outer_points, num_pts)
    inner_pc_points = sav.generate_2d_coords_for_cylinder_pca(inner_points, num_pts) #REPLACE WITH CYLINDER POINTS
    #print("outer_pc_points: ", outer_pc_points)
    #print("inner_pc_points: ", inner_pc_points)
    #Separates the x and y into their own arrays
    innerx = inner_radius_spline[:,0]
    innery = inner_radius_spline[:,1]
    outerx = outer_radius_spline[:, 0]
    outery = outer_radius_spline[:, 1]
    # concatenates them back into the format that goes into the intermediate
    inner_radius = np.concatenate((innerx, innery))
    outer_radius = np.concatenate((outerx, outery))
    #pc_points_bottom = bts.generate_2d_bottom_tissue(bts.extract_ten_coordinates_block(obj_coords_list[2])) #TODO: Errors due to not enough objects (0, 2, 1 idx should be looked at)


    # Get the PC points for Object2
    # Begin building the row to be put into the intermediate csv
    csv_row.append(file_params)  # file params

    #apex = find_apex(obj_coords_list[1])
    # apex FIX
    csv_row.extend(prop_final)
    #csv_row.append(apex)
    csv_row.extend(inner_pc_points)
    csv_row.extend(outer_pc_points)
    csv_row.extend(inner_radius)
    csv_row.extend(outer_radius)
    #csv_row.extend(pc_points_bottom)  # the 30 pc coordinates

    if first_int_file_flag:
        #TODO: Have the headers loop through the file params... Done?
        csv_header.append('File Name')
        for key, value in current_run_dict.items():
            csv_header.append(key)

        # TODO: This is purely for the coordinate headers (ADJUST 15 FOR THE MAX NUMBER OF COORDINATE HEADERS)
        for header in intermediate_Headers:
            coord = header
            for j in range(num_pts):
                csv_header.append(coord + str(j + 1))
        #TODO: commented this out because we do not have a points for 'bottom'
        """
        coord = 'Bx'
        for i in range(2):
            if i == 1:
                coord = 'By'
            for j in range(10):
                csv_header.append(coord + str(j + 1))
        """

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerow(csv_row)

    else:
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

    # sleep to give the file time to reach directory
    time.sleep(1)

    return csv_filename