"""
Model deployment
"""
import os
import json
import shutil


# Load config.json and correct path variable
config_path = "config.json"
config = json.load(open(config_path, 'r'))

model_path = os.path.join(config['output_model_path'], config['model_name'])
score_path = os.path.join(config['output_model_path'], config['score_name'])
ingest_path = os.path.join(config['output_folder_path'], config['ingestion_record'])
prod_deployment_path = config['prod_deployment_path']

os.makedirs(prod_deployment_path, exist_ok=True)


def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(model_path, prod_deployment_path)
    shutil.copy(score_path, prod_deployment_path)
    shutil.copy(ingest_path, prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()
