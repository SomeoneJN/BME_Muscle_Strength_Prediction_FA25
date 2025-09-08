import os
import re
import csv

import os
import re
import CylinderFunctions as cf
from io import StringIO

def extract_numbers_from_filename(filename):
  """
  Extracts numbers enclosed within parentheses from a filename 
  and creates a dictionary with keys like 'Part1_E', 'Part3_E', etc.

  Args:
    filename: The filename to process.

  Returns:
    A dictionary where keys are the prefixes (e.g., 'Part1_E') 
    and values are the corresponding extracted numbers.
  """
  pattern = r'(\w+_E)\(([0-9.]+)\)'
  matches = re.findall(pattern, filename)
  return {key: float(value) for key, value in matches}


def get_feb_filenames(folder_path):
  """
  Gets a list of .feb filenames within a specified folder.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of .feb filenames within the folder.
  """
  try:
    filenames = os.listdir(folder_path)
    return [f for f in filenames if f.endswith(".feb")] 
  except FileNotFoundError:
    print(f"Error: Folder not found at {folder_path}")
    return []

# def process_folder(folder_path):
#   """
#   Processes .feb files in a folder, extracts numbers from their filenames, 
#   and returns a list of dictionaries, each representing one file.

#   Args:
#     folder_path: The path to the folder.

#   Returns:
#     A list of dictionaries, where each dictionary contains extracted 
#     numbers with their corresponding prefixes as keys.
#   """
#   feb_filenames = get_feb_filenames(folder_path)
#   results = []
#   for filename in feb_filenames:
#     extracted_data = extract_numbers_from_filename(filename)
#     if extracted_data:  # Only add entries with extracted numbers
#       results.append(extracted_data)
#   return results



#################################################
def process_folder(folder_path):
  """
  Processes .feb files in a folder, extracts numbers from their filenames, 
  and creates a list of dictionaries, each representing one file.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of dictionaries, where each dictionary contains extracted 
    numbers with their corresponding prefixes as keys.
  """
  feb_filenames = get_feb_filenames(folder_path)
  results = []
  for filename in feb_filenames:
    extracted_data = extract_numbers_from_filename(filename)
    if extracted_data:  # Only add entries with extracted numbers
      results.append(extracted_data) 
  return results





# Example usage
folder_path = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing" 
extracted_data = process_folder(folder_path)

if extracted_data:
  fieldnames = list(extracted_data[0].keys()) 
else:
  fieldnames = [] 

# Print the extracted data
for row in extracted_data:
  print(row)
  

# Create an in-memory StringIO object
output = StringIO() 

# Create a DictWriter object and write data to the StringIO object
writer = csv.DictWriter(output, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(extracted_data) 

# Create a DictReader object from the StringIO object
output.seek(0)  # Reset the StringIO object's internal position to the beginning
csv_reader = csv.DictReader(output)

for row in csv_reader:
    print(row) 
    
# print(fieldnames)
# print(extracted_data.fieldnames)
  
  
dictionary_file = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing\feb_variables.csv" #DONE

run_file = open(dictionary_file)
DOE_dict = csv.DictReader(run_file)

for row in DOE_dict:
    print(row)
    
print('Fieldnames: ', DOE_dict.fieldnames)

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

"""
This file is responsible for running FeBio in conjunction with the feb_variables.csv file so that
a run is completed for each row, then is post processed
"""
import csv
import datetime
import math
import os.path
import sys
import subprocess
import xml.etree.ElementTree as ET
import PostProcess_FeBio as proc
import Bottom_Tissue_SA_Final as bts
import pandas as pd
import re
import time
import PCA_data
import ShapeAnalysisVerification as sav
from lib import IOfunctions
import CylinderFunctions
# from pygem import RBF
import numpy as np
import CylinderFunctions
import Plot_intermediate_points as pip



laptop = True


# FeBio Variables
# TODO: ENTER IN VAR FILE --> SET PART NAMES FOR ALL MATERIALS --> DONE
dictionary_file = 'C:\\Users\\EGRStudent\\PycharmProjects\\PyGem-Modifying-2023\\FeBio_Inverse\\feb_variables.csv' #DONE
FeBioLocation = 'C:\\Program Files\\FEBioStudio2\\bin\\febio4.exe'

if laptop:
    originalFebFilePath = r"C:\Users\mgordon\My Drive\a  Research\a Pelvic Floor\Inverse FEA\SU24 Analysis Code\3 Tissue Model v2.feb"
else:
    originalFebFilePath = 'D:\\Gordon\\Automate FEB Runs\\2024_5_9_NewModel\\Base_File\\3 Tissue Model v2.feb'
# Results_Folder = "D:\\Gordon\\Automate FEB Runs\\2024_12_11"
Results_Folder = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing" 
# This is for output
object_list = ['Levator Ani Side 2']
# Currently being used to access base object, may need to be changed when looking to generate multiple objects at once
part_list = ['Part1', 'Part3', 'Part7', 'Part10', 'Part11']
cylinder_parts = ['Part3']
ZeroDisplacement = "ZeroDisplacement1"
# Only necessary if running Post_Processing_Flag
numCompPCA = 3

# FLAGS
Create_New_Feb_Flag = False
Run_FeBio_File_Flag = False
first_int_file_flag = True
GENERATE_INTERMEDIATE_FLAG = True
Post_Processing_Flag = False

# PLOTTING
plot_points_on_spline = False

# TODO: Input Parameters for Cylinder Creation
num_cylinder_points = 200

#Have the default material variables be 1 (100%) so they do not change if no variable is given
# TODO: Update Everytime you want to change your base file
default_dict = {
    'Part1_E': 1,
    'Part3_E': 1,
    'Part7_E': 1,
    'Part10_E': 1,
    'Part11_E': 1,
    'Pressure': 0.015,
    'Inner_Radius': 1.25,
    'Outer_Radius': 1.75
}
default_code_dict = {
    'Part1_E': 'P1_E',
    'Part3_E': 'P3_E',
    'Part7_E': 'P7_E',
    'Part10_E': 'P10_E',
    'Part11_E': 'P11_E',
    'Pressure': 'Pre',
    'Inner_Radius': 'IR',
    'Outer_Radius': 'OR'
}

'''
Function: RunFEBinFeBio
Takes in the input feb file along with the location of FeBio on the system and runs it via 
the command line. The output log file name can also be given.
'''
def RunFEBinFeBio(inputFileName, FeBioLocation, outputFileName=None):
    CallString = '\"' + FeBioLocation + '\" -i \"' + inputFileName + '\"'
    if outputFileName is not None:
        CallString += ' -o \"' + outputFileName + '\"'
    print("CallString: ", CallString)
    subprocess.call(CallString)

'''
Function: updateProperties
Takes in a specified part name, finds the corresponding material, changes the modulus of the material, 
then saves a new input file with the name relating to the changed part (part, property, new value).

Update 4/29: Can now take in Pressure, Inner & Outer Radius for generating 3D Cylinders 
'''
def updateProperties(origFile, fileTemp):
    new_input_file = Results_Folder + '\\' + fileTemp + '.feb'
    # Parse original FEB file
    tree = ET.parse(origFile)
    root = tree.getroot()

    # Verify log file exists, if not add log file to be 'x;y;z'
    IOfunctions.checkForLogFile(root)

    # Update material property values
    for part_prop in current_run_dict.keys():
        # if it is not above names then it is a part
        if "Part" in part_prop:
            part_name = part_prop.split('_')[0]
            prop_name = part_prop.split('_')[1]

            # Locate the Mesh Domains section to find which part have which materials
            mesh_domains = root.find('MeshDomains')
            for domain in mesh_domains:
                if domain.attrib['name'] == part_name:
                    for mat in tree.find('Material'):
                        if mat.attrib['name'] == domain.attrib['mat']:
                            new_value = current_run_dict[part_prop]
                            mat.find(prop_name).text = str(new_value)

    # Update Pressure Value
    loads = root.find('Loads')
    for surface_load in loads:
        pressure = surface_load.find('pressure')
        pressure.text = str(current_run_dict["Pressure"])

    # Assign inner_radius value from "feb_variables.csv"
    final_inner_radius = float(current_run_dict["Inner_Radius"])

    # Assign outer_radius value from "feb_variables.csv"
    final_outer_radius = float(current_run_dict["Outer_Radius"])

    # Extract points from .feb file and return in array of tuples
    extract_points = CylinderFunctions.get_initial_points_from_parts(root, part_list)

    # Find cylinder height
    global cylinder_height
    cylinder_height = CylinderFunctions.findLargestZ(extract_points)

    if not math.isclose(float(current_run_dict["Inner_Radius"]), float(default_dict["Inner_Radius"]),
                        abs_tol=0.001) and not math.isclose(float(current_run_dict["Outer_Radius"]),
                                                            float(default_dict["Outer_Radius"]), abs_tol=0.001):

        # Extract only the coordinates for RBF
        initial_coordinates = np.array([coords for coords in extract_points.values()])

        # Assign initial_control_points extract_points
        initial_inner_radius, initial_outer_radius = CylinderFunctions.determineRadiiFromFEB(root, cylinder_parts)
        initial_control_points = CylinderFunctions.generate_annular_cylinder_points(initial_inner_radius,
                                                                                            initial_outer_radius,
                                                                                            cylinder_height,
                                                                                            num_cylinder_points)


        final_control_points = CylinderFunctions.generate_annular_cylinder_points(final_inner_radius, final_outer_radius,
                                                                                          cylinder_height,
                                                                                          num_cylinder_points)

        # Enter the name of surface you would like to get id's from, and it will parse the id's and append the
        # coords from those nodes to initial and final cp for rbf
        zero_displacement = np.array(CylinderFunctions.extractCoordinatesFromSurfaceName(root, ZeroDisplacement))

        initial_control_points = np.concatenate((initial_control_points, zero_displacement))
        final_control_points = np.concatenate((final_control_points, zero_displacement))

        # Call the new morph_points function
        deformed_points = CylinderFunctions.morph_points(initial_control_points, final_control_points,
                                                                 initial_coordinates,
                                                                 extract_points)
        #changes deformed points to be a dictionary
        deformed_points = {item[0]: item[1] for item in deformed_points}

        # Replace coordinates in the original file with the deformed points
        CylinderFunctions.replaceCoordinatesGivenNodeId(root, deformed_points)

    # Write the updated tree to the new FEB file
    tree.write(new_input_file, xml_declaration=True, encoding='ISO-8859-1')

    return new_input_file


# Post Processing Variables
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
obj_coords_list = []
file_num = 0
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'

#Get data from the Run_Variables file
# Newer code (2/14)


# First check to see if the FEB is to be created and run
# If create FEB, then use dictionary_file
# If no create, but run look for FEB files
# If not run, but first_int_file or Post_Processing_Flag, look for log? files

# # FLAGS
# Create_New_Feb_Flag = False
# Run_FeBio_File_Flag = False
# first_int_file_flag = True
# GENERATE_INTERMEDIATE_FLAG = True
# Post_Processing_Flag = False


if Create_New_Feb_Flag:
    run_file = open(dictionary_file)
    DOE_dict = csv.DictReader(run_file)
else:
    # DOE_dict= process_folder(Results_Folder)
    # if DOE_dict:
    #     fieldnames = list(DOE_dict[0].keys()) 
    # else:
    #     fieldnames = [] 
    
    # # Create an in-memory StringIO object
    # output = StringIO() 

    # # Create a DictWriter object and write data to the StringIO object
    # writer = csv.DictWriter(output, fieldnames=fieldnames)
    # writer.writeheader()
    # writer.writerows(DOE_dict) 

    # # Create a DictReader object from the StringIO object
    # output.seek(0)  # Reset the StringIO object's internal position to the beginning
    # csv_reader = csv.DictReader(output)
    
    
    extracted_data = process_folder(Results_Folder)
    
    if extracted_data:
      fieldnames = list(extracted_data[0].keys()) 
    else:
      fieldnames = [] 

    # # Print the extracted data
    # for row in extracted_data:
    #   print(row)
      

    # Create an in-memory StringIO object
    output = StringIO() 

    # Create a DictWriter object and write data to the StringIO object
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(extracted_data) 

    # Create a DictReader object from the StringIO object
    output.seek(0)  # Reset the StringIO object's internal position to the beginning
    DOE_dict = csv.DictReader(output)


# Example usage
folder_path = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing" 
extracted_data = process_folder(folder_path)


'''
Function: new_check_normal_run
Takes in a log file and checks for the normal termination indicator, notifying that post processing
can be done on the file
'''
def new_check_normal_run(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        last_lines = [line.strip() for line in lines[-10:]]
        for line in last_lines:
            if "N O R M A L   T E R M I N A T I O N" in line:
                return True
        return False

current_run_dict = default_dict.copy()

# Run the FEB files for each set of variables in the run_variables csv
for row in DOE_dict:
    print('Row:', row)

    # Initialize current run dictionary by changing the values of the variables
    # defined by the DOE_dict and leaving the rest as default
    for key in DOE_dict.fieldnames:
        if key in default_dict.keys():
            current_run_dict[key] = row[key]
        else:
            if key != 'Run Number':
                print(key, 'is an invalid entry')
                sys.exit()

    #generation of current run file template based on attributes
    fileTemplate = ''
    csv_template = ''

    # creating the file name base used for .feb, .log, etc.
    for key in current_run_dict:
        if float(current_run_dict[key]) != float(default_dict[key]):
            param = '' + str(key) + '(' + str(current_run_dict[key]) + ')'
            fileTemplate += param


    print("LOCATION 1")
    # Generate Log CSV File into Results Folder; this has all of the parameters for future reference
    IOfunctions.generate_log_csv(current_run_dict, default_code_dict, Results_Folder, fileTemplate + '_log' + '.csv')

    workingInputFileName = Results_Folder + '\\' + fileTemplate + '.feb'


    #######################
    # I think this is where to check to see if *.feb should be generated
    #######################

    if Create_New_Feb_Flag:
        #Update properties, create new input file
        workingInputFileName = updateProperties(originalFebFilePath, fileTemplate)

    # TODO: to easily get input files from anywhere, do the shutil move to working directory
    logFile = Results_Folder + '\\' + fileTemplate + '.log'
    # print("logFile: ", logFile)
    
    # Print Log file when flag is true
    if Run_FeBio_File_Flag:
        RunFEBinFeBio(workingInputFileName, FeBioLocation, logFile)

        # Check for success of the feb run
    if new_check_normal_run(logFile):
        # Post process
        if GENERATE_INTERMEDIATE_FLAG:
            print("Generating Intermediate File...")
            if first_int_file_flag:

                # print("Filename: ", workingInputFileName)
                # # Find the nodes that are on the edge of the cylder
                # edge_elements_dictionary = sav.getCylinderEdgePoints(workingInputFileName, cylinder_parts)
                # print("Edge Elements: ", edge_elements_dictionary)

                # # CylinderFunctions.plot_3d_points(edge_elements_dictionary)
                
                # # Find the nodes on the correct end and separate into inner and outter nodes
                # inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cylinder_height,
                #                                                    logFile, workingInputFileName, object_list[0])
                # print("IR:", inner_radius)
                # inner_radius_spline, outer_radius_spline = pip.angle_spline_driver(inner_radius, outer_radius)
                # print("fileTemplate:", fileTemplate)

                # proc.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName,
                #                        first_int_file_flag,
                #                        csv_filename, inner_radius_spline, outer_radius_spline, current_run_dict,
                #                        plot_points_on_spline)

                # first_int_file_flag = False
                
                
                # int_log_name = feb_name.split(".f")
                # int_log_name[1] = ".log"
                # log_name = int_log_name[0]+int_log_name[1]
                
                
                csv_row = []
                
                # Get the pure file name that just has the material parameters
                # file_params = int_log_name[0].split('\\')[-1]
                file_params = fileTemplate
                matches = re.findall(r'(Part\d+_E)\(([\d.]+)\)', file_params)
                current_run_dictionary = default_dict.copy()
                for part, value in matches:
                    current_run_dictionary[part] = float(value)
                print("Current_run_dictionary: ", current_run_dictionary)
                
                feb_name = workingInputFileName
                # print("feb_name:", feb_name)
                print("part list: ", part_list)
                edge_elements_dictionary = sav.getCylinderEdgePoints(feb_name, part_list)
                
                tree = ET.parse(feb_name)
                
                root = tree.getroot()
                # print(root, part_list)
                extract_points = CylinderFunctions.get_initial_points_from_parts(root, part_list)
                
                # print("extract_points:############### ", extract_points)
                # print(cylinder_height, logFile, feb_name, object_list[0])
                # print("edge_elements_dictionary", edge_elements_dictionary)
                # inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cf.findLargestZ(extract_points), log_name, feb_name, object_list[0])
                inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cf.findLargestZ(extract_points), logFile, feb_name, object_list[0])
                
                
                
                # print("##########################", inner_radius, outer_radius)
                
                inner_radius_spline, outer_radius_spline = pip.angle_spline_driver(inner_radius, outer_radius)
                
                # proc.generate_int_csvs(file_params, object_list, log_name, feb_name, first_file_flag, csv_filename, inner_radius_spline, outer_radius_spline, current_run_dictionary, plot_points_on_spline)
                proc.generate_int_csvs(file_params, object_list, logFile, feb_name, first_int_file_flag, csv_filename, inner_radius_spline, outer_radius_spline, current_run_dictionary, plot_points_on_spline)                

                # if first_file_flag:
                #     first_file_flag = False
                
                if first_int_file_flag:
                    first_int_file_flag = False
                
                # sleep to give the file time to reach directory
                time.sleep(1)
                file_num += 1
                print(str(file_num) + ": " + file_params)
                obj_coords_list = []
                

            else:

                edge_elements_dictionary = sav.getCylinderEdgePoints(workingInputFileName, cylinder_parts)

                inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cylinder_height,
                                                                    logFile, workingInputFileName, object_list[0])

                inner_radius_spline, outer_radius_spline = pip.angle_spline_driver(inner_radius, outer_radius)

                proc.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName,
                                        first_int_file_flag,
                                        csv_filename, inner_radius_spline, outer_radius_spline, current_run_dict,
                                        plot_points_on_spline)

        file_num += 1
        print('Completed Iteration ' + str(file_num) + ": " + fileTemplate)
        obj_coords_list = []

    else:
        os.rename(workingInputFileName, os.path.splitext(workingInputFileName)[0] + '_error.feb')

if Post_Processing_Flag:  # previously called final_csv_flag
    proc.process_features(csv_filename, Results_Folder, date_prefix, numCompPCA, current_run_dict)
