"""
Model Training

Author: quangdungluong
Date: February 21, 2023
"""
import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load config.json and get path variables
config_path = "config.json"
config = json.load(open(config_path, 'r'))

dataset_csv_path = os.path.join(config['output_folder_path'], config['final_data'])
model_path = os.path.join(config['output_model_path'], config['model_name'])

# Create output model path
os.makedirs(config['output_model_path'], exist_ok=True)


def train_model():
    # Function for training the model
    data = pd.read_csv(dataset_csv_path)

    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target = ["exited"]
    x_train = data[features]
    y_train = data[target]

    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    model.fit(x_train, y_train)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(model_path, 'wb'))


if __name__ == "__main__":
    train_model()
