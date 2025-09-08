import datetime
import xml.etree.ElementTree as ET
import time
import glob
import PostProcess_FeBio as proc
import ShapeAnalysisVerification as sav
import CylinderFunctions
import CylinderFunctions as cf
import Plot_intermediate_points as pip
import re
"""
The Purpose of this file is to allow us to manually test the generation of final modified train csv files from the given 
intermediate csv and determine if we are are acquiring our PCA Points Correctly
"""

# TODO: Change these parameters
current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)
Results_Folder ="C:\\Users\\mgordon\\My Drive\\a  Research\\a Pelvic Floor\\Inverse FEA\\SU24 Analysis Code\\Testing_Results\\" # INTERMEDIATE CSV ENDS UP HERE
Target_Folder = "C:\\Users\\mgordon\\My Drive\\a  Research\\a Pelvic Floor\\Inverse FEA\\SU24 Analysis Code\\Runs for Testing\\*.feb"  # LOOK HERE FOR THE FEB FILES

# Enter CSV File name manually or have set to date prefix when generating new intermediate
csv_filename = Results_Folder + '\\' + date_prefix + '_intermediate.csv'
#csv_filename = 'D:\\Gordon\\Automate FEB Runs\\2024_10_28\\2024_10_29_intermediate.csv'

object_list = ['Levator Ani Side 2']  # MAKE SURE THIS MATCHES THE OBJECTS IN THE CURRENTLY USED MODEL
part_list = ['Part1', 'Part3', 'Part7', 'Part10', 'Part11']
obj_coords_list = []
file_num = 0
numCompPCA = 3

# TODO: Change Flags to adjust processing
first_file_flag = True
GENERATE_INTERMEDIATE_FLAG = True
final_csv_flag = False
plot_points_on_spline = False

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

if GENERATE_INTERMEDIATE_FLAG:

    for feb_name in glob.glob(Target_Folder):

        int_log_name = feb_name.split(".f")
        int_log_name[1] = ".log"
        log_name = int_log_name[0]+int_log_name[1]
        # print("log_name: ", log_name)

        csv_row = []

        # Get the pure file name that just has the material parameters
        file_params = int_log_name[0].split('\\')[-1]
        matches = re.findall(r'(Part\d+_E)\(([\d.]+)\)', file_params)
        current_run_dictionary = default_dict.copy()
        for part, value in matches:
            current_run_dictionary[part] = float(value)
        print("Current_run_dictionary: ", current_run_dictionary)

        # print("feb_name: ", feb_name)
        print("part list: ", part_list)
        edge_elements_dictionary = sav.getCylinderEdgePoints(feb_name, part_list)

        tree = ET.parse(feb_name)

        root = tree.getroot()
        # print(root, part_list)
        extract_points = CylinderFunctions.get_initial_points_from_parts(root, part_list)
        
        # print("extract_points:############### ", extract_points)

        cylinder_height = cf.findLargestZ(extract_points)
        logFile = log_name
        print(cylinder_height, logFile, feb_name, object_list[0])
        # inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cf.findLargestZ(extract_points), log_name, feb_name, object_list[0])
        inner_radius, outer_radius = sav.getRadiiFromEdges(edge_elements_dictionary, cylinder_height, logFile, feb_name, object_list[0])

        # print("edge_elements_dictionary", edge_elements_dictionary)
        # print("##########################", inner_radius, outer_radius)
        # inner_radius_spline, outer_radius_spline = pip.angle_spline_driver(inner_radius, outer_radius)
        
        # print("file_params:", file_params)
        # print("feb_name:", feb_name)
        # proc.generate_int_csvs(file_params, object_list, log_name, feb_name, first_file_flag, csv_filename, inner_radius_spline, outer_radius_spline, current_run_dictionary, plot_points_on_spline)

        # if first_file_flag:
        #     first_file_flag = False

        # # sleep to give the file time to reach directory
        # time.sleep(1)
        # file_num += 1
        # print(str(file_num) + ": " + file_params)
        # obj_coords_list = []


if final_csv_flag:
    print('Generating PCA File')
    filepath = proc.process_features(csv_filename, Results_Folder, date_prefix, numCompPCA)
    print("File path: ", filepath)