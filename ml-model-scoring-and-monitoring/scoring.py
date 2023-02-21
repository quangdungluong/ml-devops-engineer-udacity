"""
Model Scoring

Author: quangdungluong
Date: February 21, 2023
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, session
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load config.json and get input and output paths
config_path = "config.json"
config = json.load(open(config_path, 'r'))

dataset_csv_path = os.path.join(config['output_folder_path']) 
dataset_name = "testdata.csv"
test_data_path = os.path.join(config['test_data_path'], dataset_name) 
model_name = "trainedmodel.pkl"
output_model_path = os.path.join(config['output_model_path'], model_name)
score_name = "latestscore.txt"
score_path = os.path.join(config['output_model_path'], score_name)

# Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_data = pd.read_csv(test_data_path)
    model = pickle.load(open(output_model_path, 'rb'))

    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target = ["exited"]
    x_test = test_data[features].values.reshape(-1, len(features))
    y_test = test_data[target].values.reshape(-1, len(target)).ravel()

    prediction = model.predict(x_test)
    f1 = f1_score(prediction, y_test)

    with open(score_path, 'w') as f:
        f.write(str(f1))

if __name__ == "__main__":
    score_model()