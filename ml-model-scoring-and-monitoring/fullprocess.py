"""
Process Automation

Author: quangdungluong
Date: February 22, 2023
"""
import json
import os
from scoring import score_model
import pickle
import pandas as pd
import subprocess

# Read config
config_path = "config.json"
config = json.load(open(config_path, 'r'))


def check_new_data():
    # Check and read new data
    # Read ingestedfiles.txt
    ingested_file_path = os.path.join(config['prod_deployment_path'], config['ingestion_record'])
    with open(ingested_file_path, 'r') as f:
        ingested_files = f.read().splitlines()
    # Check source directory
    source_files = []
    for file_path in os.listdir(config['input_folder_path']):
        if file_path.endswith('.csv'):
            source_files.append(file_path)
    # Do ingestion if needed
    if ingested_files != source_files:
        print(source_files, ingested_files)
        subprocess.call(["python", "ingestion.py"])
        return True
    return False


def check_model_drift():
    # Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    latest_score_path = os.path.join(config['prod_deployment_path'], config['score_name'])
    latest_score = float(open(latest_score_path, 'r').read())

    model = pickle.load(open(os.path.join(config['prod_deployment_path'], config['model_name']), 'rb'))
    recent_data_path = os.path.join(config['output_folder_path'], config['final_data'])
    recent_data = pd.read_csv(recent_data_path)
    
    recent_score = score_model(model=model, test_data=recent_data)

    print(recent_score, latest_score)
    return recent_score < latest_score


if __name__ == "__main__":
    if check_new_data():
        if check_model_drift():
            print("Model drift")
            subprocess.call(["python", "training.py"])
            subprocess.call(["python", "scoring.py"])
            subprocess.call(["python", "deployment.py"])

            subprocess.call(["python", "diagnostics.py"])
            subprocess.call(["python", "reporting.py"])
            subprocess.call(["python", "apicalls.py"])
