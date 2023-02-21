"""
Model deployment
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


# Load config.json and correct path variable
config_path = "config.json"
config = json.load(open(config_path, 'r'))

model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")
score_path = os.path.join(config['output_model_path'], "latestscore.txt")
ingest_path = os.path.join(config['output_folder_path'], "ingestedfiles.txt")
prod_deployment_path = config['prod_deployment_path']

os.makedirs(prod_deployment_path, exist_ok=True)
####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(model_path, prod_deployment_path)
    shutil.copy(score_path, prod_deployment_path)
    shutil.copy(ingest_path, prod_deployment_path)

if __name__ == "__main__":
    store_model_into_pickle()
