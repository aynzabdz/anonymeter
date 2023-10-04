import pandas as pd
import numpy as np
import os
import json


def strip_strings(s):
    
    """
    Removes leading and trailing whitespaces from a string. 
    If the provided object is not a string, returns it unchanged.
    
    Parameters:
        s (str or object): Input string or object.
        
    Returns:
        str or object: Trimmed string or original object.
    """
    
    
    if isinstance(s, str):
        return s.strip()
    return s


def de_anonymize_column(df_anonymized, df_original, column):
    
    """
    De-anonymize a specific column of a dataframe using a mapping hierarchy.
    
    Parameters:
        df_anonymized (pandas.DataFrame): Dataframe with anonymized values.
        df_original (pandas.DataFrame): Original dataframe with real values.
        column (str): Column name to be de-anonymized.
        
    Returns:
        pandas.DataFrame: Dataframe with the specified column de-anonymized.
    """
    
    df_anonymized = df_anonymized.applymap(strip_strings)
    df_anonymized.replace('?', np.nan, inplace=True)
    df_anonymized = df_anonymized.dropna()
    
    hierarchy_filepath = os.path.join(config_path, hierarchy_files[column])
    hierarchy_file = pd.read_csv(hierarchy_filepath)
             
    mapping_dict = {}
    for idx, row in hierarchy_file.iterrows():
        for value in row:
            if value not in mapping_dict:
                mapping_dict[value] = set()
            mapping_dict[value].add(row[0])
            
    unique_anonymized_values = df_anonymized[column].unique()
    
    for anonymized_value in unique_anonymized_values:
                        
        possible_deanonymized_values = list(mapping_dict.get(anonymized_value, []))
        if not possible_deanonymized_values:
            print(f"No possible deanonymized values for {anonymized_value}")
            continue
        
        value_counts = df_original[df_original[column].isin(possible_deanonymized_values)][column].value_counts(normalize=True)
        
        if value_counts.empty:
            print(f"Value counts empty for {anonymized_value}")
            continue
        
        mask = df_anonymized[column] == anonymized_value
        num_rows_to_change = mask.sum()

        if num_rows_to_change == 0:
            print(f"No rows to change for {anonymized_value}")
            continue
        
        replacement_values = np.random.choice(value_counts.index, p=value_counts.values, size=num_rows_to_change)
        df_anonymized.loc[mask, column] = replacement_values

    return df_anonymized


def de_anonymize(df_anonymized, df_original):
    
    """
    De-anonymize multiple columns of a dataframe using corresponding mapping hierarchies.
    
    Parameters:
        df_anonymized (pandas.DataFrame): Dataframe with anonymized values.
        df_original (pandas.DataFrame): Original dataframe with real values.
        
    Returns:
        pandas.DataFrame: Dataframe with multiple columns de-anonymized.
    """
    
    for column_name in ['age', 'education', 'marital', 'race']:
        df_anonymized = de_anonymize_column(df_anonymized, df_original, column_name)
    return df_anonymized



with open('deanonymization_config.json', 'r') as config_file:
    config = json.load(config_file)

data_path = config['data_path']
original_data_file = config['files']['original_data']
config_path = config['config_path']
hierarchy_files = config['files']['hierarchy_files']