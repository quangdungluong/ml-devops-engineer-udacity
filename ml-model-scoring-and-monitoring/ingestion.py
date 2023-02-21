"""
Data Ingestion

Author: quangdungluong
Date: February 21, 2023
"""
import json
import os

import pandas as pd

# Load config.json and get input and output paths
config_path = "config.json"
config = json.load(open(config_path, 'r'))

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_file_name = "finaldata.csv"
record_file_name = "ingestedfiles.txt"

# Creating output folder path
os.makedirs(output_folder_path, exist_ok=True)

# Function for data ingestion
def merge_multiple_dataframe():
    # Reading data and compiling a dataset
    main_df = pd.DataFrame()
    records = []
    for file in os.listdir(input_folder_path):
        if file.endswith(".csv"):
            curr_df = pd.read_csv(os.path.join(input_folder_path, file))
            main_df = main_df.append(curr_df)
            records.append(file)

    # Drop duplicates
    main_df = main_df.drop_duplicates(ignore_index=True)
    # Writing the dataset
    main_df.to_csv(os.path.join(output_folder_path, output_file_name))
    # Save a record of the ingestion
    with open(os.path.join(output_folder_path, record_file_name), 'w') as f:
        for record in records:
            f.write(str(record) + '\n')


if __name__ == '__main__':
    merge_multiple_dataframe()
