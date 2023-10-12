import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from de_anonymize import de_anonymize

def parse_args():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser(description='Process anonymization flags.')
    parser.add_argument('--anonymize', action='store_true', default=False, help='Flag indicating if data should be anonymized.')
    parser.add_argument('--k', type=int, required=False, help='Value of k for k-anonymization. Required if --anonymize is provided.')
    
    return parser.parse_args()

def load_data(config):
    """
    Load datasets from the specified paths in the configuration.

    Args:
        config (dict): Configuration dictionary with data paths.

    Returns:
        tuple: Tuple containing original dataframe and synthetic dataframe.
    """
    ori_path = os.path.join(config["data_path"], config["ori_file"])
    syn_path = os.path.join(config["data_path"], config["syn_file"])
    return pd.read_csv(ori_path), pd.read_csv(syn_path)

def anonymize_data(args, config, df_ori):
    """
    Anonymize the original data if the anonymize flag is set.

    Args:
        args (argparse.Namespace): Command-line arguments.
        config (dict): Configuration dictionary.
        df_ori (pd.DataFrame): Original dataframe to be anonymized.

    Returns:
        pd.DataFrame: Anonymized dataframe.
    """
    if args.anonymize:
        if args.k is None:
            raise ValueError("--k is required if --anonymize is provided.")
        script_name = "./k_anonymize.py"
        result = subprocess.run(["python", script_name, '--K', str(args.k)])
        if result.returncode != 0:
            raise Exception("Error in subprocess: k_anonymize.py")
        syn_filename = f"{config['syn_file'].replace('.csv', '')}_{args.k}_anonymized.csv"
        syn_path = os.path.join(config["data_path"], syn_filename)
        df_syn = pd.read_csv(syn_path)
        return de_anonymize(df_syn, df_ori)
    return None

def prepare_data(df_ori):
    """
    Prepare the data for model training by splitting it into train and test sets.

    Args:
        df_ori (pd.DataFrame): Original dataframe.

    Returns:
        tuple: Tuple containing train-test split of data.
    """
    target_column = "income"
    y = df_ori[target_column].replace({'<=50K': 0, '>50K': 1})
    X = df_ori.drop(target_column, axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_pipeline(X):
    """
    Create a pipeline with one-hot encoding step steps and a classifier.

    Args:
        X (pd.DataFrame): Training data to infer the categorical columns.

    Returns:
        Pipeline: Scikit-learn pipeline.
    """
    cat_cols = [col for col in X.columns if X[col].dtype == 'object']
    transformer = ColumnTransformer([("encoder", OneHotEncoder(), cat_cols)], remainder='passthrough')
    clf = RandomForestClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=2, n_estimators=50, random_state=42)
    return Pipeline([('transformer', transformer), ('classifier', clf)])



def compute_utility(anonymize=False, k=None):
    """
    Compute utility by executing the pipeline: 
    - Load data
    - Optionally anonymize data
    - Prepare data
    - Train a model
    - Evaluate model on (anonymized) synthetic data
    """
    class Args:
        pass

    args = Args()
    args.anonymize = anonymize
    args.k = k
    
    with open("config.json", "r") as f:
        config = json.load(f)

    df_ori, df_syn = load_data(config)
    if args.anonymize:
        df_syn = anonymize_data(args, config, df_ori)
    
    X_train, X_test, y_train, y_test = prepare_data(df_ori)
    pipeline = get_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_syn = df_syn["income"].replace({'<=50K': 0, '>50K': 1})
    X_syn = df_syn.drop("income", axis=1)
    y_proba_syn = pipeline.predict_proba(X_syn)[:, 1]
    y_pred_syn = (y_proba_syn > 0.5).astype(int)
    
    accuracy_syn = accuracy_score(y_syn, y_pred_syn)
    f1_syn = f1_score(y_syn, y_pred_syn, average='weighted')
    roc_auc_syn = roc_auc_score(y_syn, y_proba_syn)

    dataset_name = "Synthetic Data"
    if args.anonymize:
        dataset_name += f" ({args.k}{' ' if args.k is not None else ''}anonymized)"
    
    print(f"\n------------- {dataset_name} -------------")
    
    print(f"Accuracy on Synthetic Data: {accuracy_syn}")
    print(f"F1 Score on Synthetic Data: {f1_syn}")
    print(f"ROC AUC on Synthetic Data: {roc_auc_syn}")
    
    return accuracy_syn, f1_syn, roc_auc_syn



def compute_test_utility():
    """
    Compute utility metrics for the test data.

    Returns:
        tuple: Tuple containing accuracy, F1 score, and ROC AUC score for test data.
    """
    
    with open("config.json", "r") as f:
        config = json.load(f)
    
    df_ori, _ = load_data(config)
    X_train, X_test, y_train, y_test = prepare_data(df_ori)
    pipeline = get_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test > 0.5).astype(int)
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    roc_auc_test = roc_auc_score(y_test, y_proba_test)
    
    print(f"\n------------- Test Data -------------")
    print(f"\nAccuracy on Test Data: {accuracy_test}")
    print(f"F1 Score on Test Data: {f1_test}")
    print(f"ROC AUC on Test Data: {roc_auc_test}")
    
    return accuracy_test, f1_test, roc_auc_test



def optimize_rf_params(data_path):
    """
    Conduct a grid search to find the optimal parameters for the RandomForestClassifier
    within a given pipeline.
    

    Args:
    - data_path (str): The path to the dataset to be used for grid search.

    Returns:
    - dict: Best parameters for the RandomForestClassifier.
    
    Usage:
    This function was used to find the optimal model parameters of RandomForestClassifier
    . To reproduce the results, uncomment and run this function.
    """
    
    # Load and prepare data
    df_ori = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df_ori)
    pipeline = get_pipeline(X_train)
    
    print("performing grid search")

    # Define parameter grid to search
    param_grid = {
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30, 40],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)

    # Conduct the grid search on the provided data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters for RandomForestClassifier:", best_params)
    
    return best_params

# To reproduce the results, uncomment the following line:
# optimize_rf_params('../../data/adults_train.csv')



if __name__ == "__main__":
    args = parse_args()
    # accuracy, f1, roc_auc = compute_utility(args.anonymize, args.k)
