"""
Generating Plots

Author: quangdungluong
Date: February 22, 2023
"""
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


# Load config.json and get path variables
config_path = "config.json"
config = json.load(open(config_path, 'r'))

dataset_csv_path = os.path.join(config['test_data_path'], config['test_data'])


# Function for reporting
def generate_plots():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    data = pd.read_csv(dataset_csv_path)
    predictions = model_predictions(data_path=dataset_csv_path)
    ground_truth = data['exited'].values.reshape(-1, 1)
    cm = confusion_matrix(ground_truth, predictions)
    df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(config['output_model_path'], config['confusion_matrix']))

if __name__ == '__main__':
    generate_plots()
