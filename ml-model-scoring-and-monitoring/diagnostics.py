"""
Model and Data Diagnostics

Author: quangdungluong
Date: February 21, 2023
"""
import json
import os
import pickle
import subprocess
import timeit

import pandas as pd

from ingestion import merge_multiple_dataframe
from training import train_model

# Load config.json and get environment variables
config_path = "config.json"
config = json.load(open(config_path, 'r'))

dataset_csv_path = os.path.join(config['output_folder_path'], config['final_data'])
test_data_path = os.path.join(config['test_data_path'], config['test_data'])

model_path = os.path.join(config['prod_deployment_path'], config['model_name'])


def model_predictions(data_path=test_data_path):
    # Function to get model predictions
    # read the deployed model and a test dataset, calculate predictions
    data = pd.read_csv(data_path)
    model = pickle.load(open(model_path, 'rb'))
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    data = data[features].values.reshape(-1, len(features))
    pred = model.predict(data)
    return pred.tolist()


def dataframe_summary(data_path=dataset_csv_path):
    # Function to get summary statistics
    # calculate summary statistics here
    data = pd.read_csv(data_path)
    numeric_columns = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited']
    data_stats = data.describe().loc[['mean', 'std'], numeric_columns]
    median_list = [data[feature].median(axis=0) for feature in numeric_columns]
    median_stat = pd.DataFrame([median_list], columns=numeric_columns, index=['median'])
    data_stats = pd.concat([data_stats, median_stat])
    return data_stats


def missing_data(data_path=dataset_csv_path):
    data = pd.read_csv(data_path)
    na_list = list(data.isna().sum(axis=0))
    na_percents = [na_list[i]/len(data.index) for i in range(len(na_list))]
    return na_percents


def execution_time():
    # Function to get timings
    # merge_multiple_dataframe execution time
    start_time = timeit.default_timer()
    merge_multiple_dataframe()
    merge_time = timeit.default_timer() - start_time
    # train_model execution time
    start_time = timeit.default_timer()
    train_model()
    train_time = timeit.default_timer() - start_time
    return [merge_time, train_time]


def outdated_packages_list():
    # Function to check dependencies
    packages_dict = {"module_name": [],
                     "current_version": [],
                     "latest_version": []}
    with open("requirements.txt", 'r') as f:
        modules = f.read().splitlines()
    for module in modules:
        module_name, current_version = module.split('==')
        reqs = subprocess.check_output(['pip', 'index', 'versions', module_name]).decode('utf-8')
        latest_version = reqs.split('LATEST:')[1].strip()
        packages_dict['module_name'].append(module_name)
        packages_dict['current_version'].append(current_version)
        packages_dict['latest_version'].append(latest_version)
    df = pd.DataFrame.from_dict(packages_dict)
    return df


if __name__ == '__main__':
    print("Model prediction:", model_predictions(test_data_path))
    print("Summary Statistics:", dataframe_summary())
    print("Missing data:", missing_data())
    print("Data ingestion time: {:.3f}, Model training time: {:.3f}".format(*execution_time()))
    print("Dependencies:\n", outdated_packages_list())
