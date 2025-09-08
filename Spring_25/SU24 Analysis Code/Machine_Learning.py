import predict_functions as pf
import PostProcess_FeBio as pp
import PCA_data as pd
import datetime
import random

current_date = datetime.datetime.now()
date_prefix = str(current_date.year) + '_' + str(current_date.month)  + '_' + str(current_date.day)

# TODO: Adjust parameters for desired outcome
epochs = 70
epochs_start = 0
layers = 2
capacity = 96
patience = 7 # originally 50
numCompPCA = 3
squared = False
random.seed(33)
# This is the intermediate file which will be used to run process_features and used for machine learning
# data = "D:\\Gordon\\Automate FEB Runs\\2024_11_13\\2024_11_21_intermediate_chopped.csv"
data = "D:\\Gordon\\Automate FEB Runs\\2024_11_13\\2024_11_21_intermediate.csv"


# Flags
generate_PCs = True
machine_learning = True

def machine_learning_all(epochs, epochs_start, layers, capacity, patience, squared, data=None):
    # TODO: Step 1 - generate the csv file (all)
    # csv_all = pp.generate_int_csvs(fileTemplate, object_list, logFile, workingInputFileName, first_int_file_flag, csv_filename)


    # TODO: STEP 2 - load dataset
    train_data_path, test_data_path = pf.generate_train_test_csvs_files_from(data)
    print("train_data_path", train_data_path)
    print("test_data_path", test_data_path)

    # TODO: STEP 3 - Generate new PCs based on the train file
    Results_Folder = "NewMLSystem"
    mod_train_path, pca1, pcaB, pcaR = pp.process_features(train_data_path, Results_Folder, date_prefix, numCompPCA)
    print("mod_train_path", mod_train_path)

    # Save PCs and save the PCA Model
    pc_csv_path = pf.generate_PC_csv_file_from(mod_train_path, [5, 6, 7, 8])
    print("pc_csv_path",pc_csv_path)


    # add noise to test file
    mod_test_path, pca1, pcaB, pcaR = pp.process_features(test_data_path, Results_Folder, date_prefix, numCompPCA)
    print("mod_test_path: ", mod_test_path)
    #todo:Does not work
    noise_test_path = pd.add_noise_to_csv(test_data_path, Results_Folder, pca1, pcaB, noise_scale=0)
    #todo:Does not work



    # We're currently not using the saved PC models from the train to do the test, so I will create them together
    # Otherwise they will be using different models






    # TEST USE!!!
    # test = pf.generate_train_test_csvs_files_from(mod_test_path)[1]

    # TODO: Step

    if machine_learning:
        model_path, learning_curve_path, predict_directory, predicted_path, is_path = pf.machine_learning_save_predict(mod_train_path, mod_test_path, epochs, layers, capacity, patience, epochs_start)

    # TODO: Step 6 - analysis and

    # pf.write_predicted_y_to_csv(output_predicted, predicted_y)
    # pf.write_predicted_y_analysed_to_csv(output_analysis, predicted_y, )

    # print("output predicted name: ", output_predicted.title())
    # print("output analysis name: ", output_analysis.title())
    # print("output learning curve: ", output_image.title())
    print("output train data: ", train_data_path)
    print("output test data: ", test_data_path)
    # print("Modified train data: ", mod_train_path)
    # print("Modified test data: ", mod_test_path)
    # print("Saved PCs data: ", pc_csv_path)
    # print("noise_0 test data: ", noise_test_path)
    # print("Saved best model name: ", model_path)
    # print("Model learning curve: ", learning_curve_path)
    # print("Predicted data folder: ", predict_directory)
    # print("Predicted data file name: ", predicted_path)
    # print("Analysed data file name: ", analysis_path)


machine_learning_all(epochs, epochs_start, layers, capacity, patience, squared, data)
# # pf.machine_learning_save_predict(train_data, test_data)