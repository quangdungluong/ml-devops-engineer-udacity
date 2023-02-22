"""
Model Scoring

Author: quangdungluong
Date: February 21, 2023
"""
import json
import os
import pickle

import pandas as pd
from sklearn.metrics import f1_score

# Load config.json and get input and output paths
config_path = "config.json"
config = json.load(open(config_path, 'r'))

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], config['test_data'])
output_model_path = os.path.join(config['output_model_path'], config['model_name'])
score_path = os.path.join(config['output_model_path'], config['score_name'])


def score_model(model, test_data):
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target = ["exited"]
    x_test = test_data[features]
    y_test = test_data[target]

    prediction = model.predict(x_test)
    f1 = f1_score(y_test, prediction)

    with open(score_path, 'w') as f:
        f.write(str(f1))
    return f1


if __name__ == "__main__":
    test_data = pd.read_csv(test_data_path)
    model = pickle.load(open(output_model_path, 'rb'))
    print(score_model(model=model, test_data=test_data))
