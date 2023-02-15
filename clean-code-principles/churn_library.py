"""
A library of functions to find customers who are likely to churn.

Author: quangdungluong
Date: February 15, 2023
"""


import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Churn distribution
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 15))
    df['Churn'].hist()
    plt.savefig("./images/eda/churn_distribution.png")
    plt.close()

    # Customer age distribution
    plt.figure(figsize=(20, 15))
    df['Customer_Age'].hist()
    plt.savefig("./images/eda/customer_age_distribution.png")
    plt.close()

    # Marial status distribution
    plt.figure(figsize=(20, 15))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/marital_status_distribution.png")
    plt.close()

    # Total transaction distribution
    plt.figure(figsize=(20, 15))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("./images/eda/total_transaction_distribution.png")
    plt.close()

    # Plot heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/heatmap.png")
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        feature_lst = []
        feature_groups = df.groupby(feature).mean()[response]
        for val in df[feature]:
            feature_lst.append(feature_groups.loc[val])
        df[f'{feature}_{response}'] = feature_lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    category_lst = ['Gender', 'Education_Level',
                    'Marital_Status', 'Income_Category', 'Card_Category']
    df = encoder_helper(df, category_lst, response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    x_data = pd.DataFrame()
    x_data[keep_cols] = df[keep_cols]
    y_data = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''
    # Random Forest
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/rf_results.png")
    plt.close()

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/logistic_results.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                     None
    '''
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    output:
                      None
    '''
    # Initialize model
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Training model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Plot ROC_curve
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.close()
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig("./images/results/roc_curve_result.png")
    plt.close()

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    # Classification Report
    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    # Calculate feature importances
    feature_importance_plot(model=cv_rfc.best_estimator_, X_data=X_train,
                            output_pth="./images/results/feature_importances.png")


if __name__ == "__main__":
    DATA_PATH = "./data/bank_data.csv"
    # Read data
    df = import_data(pth=DATA_PATH)
    # Perform EDA and plot visualizations
    perform_eda(df=df)
    # Process the dataframe
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    # Train and store model results
    train_models(x_train, x_test, y_train, y_test)
