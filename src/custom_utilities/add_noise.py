import numpy as np
import pandas as pd

def add_noise(df, noise_level, target=None):
    """
    Adds specified noise to a given DataFrame, excluding the target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to which noise should be added.
    - noise_level (str): The level of noise to be added. It can be 'low', 'medium', or 'high'.
    - target (str, optional): The target column name which should be excluded from noise addition. 
                              If not provided, the last column of the DataFrame is considered as the target.

    Returns:
    - pd.DataFrame: A new DataFrame with added noise.

    Note:
    - For numeric columns, Gaussian noise is added based on the specified noise level.
    - For categorical columns, values are swapped based on the specified noise probability.
    """

    gaussian_std = {'low': 0.1, 'medium': 0.5, 'high': 1.0}
    categorical_prob = {'low': 0.05, 'medium': 0.15, 'high': 0.3}
    
    if not target:
        target = df.columns[-1]
    
    noisy_df = df.copy()
    
    for col in df.columns:
        if col != target:  # Ignore the target column
            if df[col].dtype in ['int64', 'float64']:
                noise = np.random.normal(0, gaussian_std[noise_level], df[col].shape)
                noisy_df[col] += noise
            else:
                swap = df[col].sample(frac=categorical_prob[noise_level]).index
                alt_values = df[col].drop(index=swap).sample(len(swap)).values
                noisy_df.loc[swap, col] = alt_values
                
    return noisy_df
