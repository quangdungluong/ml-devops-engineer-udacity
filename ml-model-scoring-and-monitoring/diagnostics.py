"""
Model and Data Diagnostics

Author: quangdungluong
Date: February 21, 2023
"""
import pickle
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

from training import train_model
from ingestion import merge_multiple_dataframe

##################Load config.json and get environment variables
config_path = "config.json"
config = json.load(open(config_path, 'r'))

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")
test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
test_data = pd.read_csv(test_data_path)

model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')

# Function to get model predictions
def model_predictions(data: pd.DataFrame):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(model_path, 'rb'))
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    data = data[features].values.reshape(-1, len(features))
    pred = model.predict(data)
    return pred.tolist()

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    
    return #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    # merge_multiple_dataframe execution time
    start_time = timeit.default_timer()
    merge_multiple_dataframe()
    merge_time = timeit.default_timer() - start_time
    # train_model execution time
    start_time = timeit.default_timer()
    train_model()
    train_time = timeit.default_timer() - start_time
    return [merge_time, train_time]

##################Function to check dependencies
def outdated_packages_list():
    broken = subprocess.check_output(['pip', 'check'])
    print(broken)

if __name__ == '__main__':
    print("Model prediction:", model_predictions(test_data))
    print("Summary Statistics:", dataframe_summary())
    print("Data ingestion time: {}, Model training time: {}".format(*execution_time()))
    outdated_packages_list()





    
