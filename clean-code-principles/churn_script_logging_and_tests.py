"""
Contain unit tests for the churn_library.py functions.

Author: quangdungluong
Date: February 15, 2023
"""
import logging
import sys

import joblib
import pytest

import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def df_raw():
    """Create raw df fixture"""
    try:
        dataframe = cl.import_data("./data/bank_data.csv")
        logging.info("Create raw data fixture: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err
    return dataframe


@pytest.fixture(scope="module")
def df_encoded(df_raw):
    """Create encoded df fixture"""
    category_lst = ['Gender', 'Education_Level',
                    'Marital_Status', 'Income_Category', 'Card_Category']
    try:
        dataframe = cl.encoder_helper(df=df_raw, category_lst=category_lst, response="Churn")
        logging.info("Create encoded df fixture: SUCCESS")
    except KeyError as err:
        logging.info("Testing encoder_helper: Columns to encode don't exist")
        raise err
    return dataframe


@pytest.fixture(scope="module")
def df_feat_eng(df_raw):
    """Create feature engineering df fixture"""
    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(df_raw, "Churn")
        logging.info("Create feature engineering df fixture: SUCCESS")
    except BaseException as err:
        logging.error("Testing perform_feature_engineering: Keep columns don't exist")
        raise err
    return x_train, x_test, y_train, y_test


def test_import(df_raw):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        assert df_raw.shape[0] > 0
        assert df_raw.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_raw):
    '''
    test perform eda function
    '''
    cl.perform_eda(df=df_raw)

    # Check results after the perform_eda has been run
    try:
        for image_name in ["churn_distribution", "customer_age_distribution",
                           "marital_status_distribution", "total_transaction_distribution",
                           "heatmap"]:
            with open(f"./images/eda/{image_name}.png") as _:
                # Check file exists
                pass
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: EDA figures missing")
        raise err


def test_encoder_helper(df_encoded):
    '''
    test encoder helper
    '''
    try:
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing encoder_helper: The file doesn't appear to have rows and columns")
        raise err

    category_lst = ['Gender', 'Education_Level',
                    'Marital_Status', 'Income_Category', 'Card_Category']
    category_lst = [category + "_Churn" for category in category_lst]
    try:
        for category in category_lst:
            assert category in df_encoded.columns
    except AssertionError as err:
        logging.error("Testing encoder_helper: The df doesn't have the right encoded columns.")
        raise err

    logging.info("Testing encoder_helper: SUCCESS")
    return df_encoded


def test_perform_feature_engineering(df_feat_eng):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train = df_feat_eng[0]
        x_test = df_feat_eng[1]
        y_train = df_feat_eng[2]
        y_test = df_feat_eng[3]
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert x_train.shape[1] == x_test.shape[1]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: ")
        raise err


def test_train_models(df_feat_eng):
    '''
    test train_models
    '''
    cl.train_models(df_feat_eng[0], df_feat_eng[1], df_feat_eng[2], df_feat_eng[3])
    try:
        joblib.load("./models/rfc_model.pkl")
        joblib.load("./models/logistic_model.pkl")
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The models weights not found")
        raise err

    try:
        for image_name in ["feature_importances", "logistic_results",
                           "rf_results", "roc_curve_result"]:
            with open(f"./images/results/{image_name}.png", 'r') as _:
                # Check whether exists
                pass
        logging.info("Testing generate_report: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing generate_report: Some reports not found")
        raise err


if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "churn_script_logging_and_tests.py"]))
