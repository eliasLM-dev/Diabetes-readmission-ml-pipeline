""""
This module contains different, smaller, functions that are used in the main pipeline. The functions
are designed to be modular and reusable, allowing for easy integration into different parts of the pipeline.
The functions are organized into different categories based on their functionality, such as data cleaning, data transformation,
data analysis, and data visualization. Each function is documented with a docstring that describes its purpose, input parameters, and return values.
The functions are designed to be efficient and optimized for performance, using vectorized operations and avoiding loops where possible.
"""
# Importing necessary libraries
import pandas as pd
import numpy as np

# ---------------   Data Cleaning Functions   ---------------
def placeholder_check_and_drop(df, column_name, placeholder_codes, threshold):
    """
    Checks the total proportion of placeholder codes in a column.
    Drops rows containing them if total is below threshold.
    Otherwise, replaces those codes with NaN.
    
    Args:
        df (DataFrame): The DataFrame to operate on.
        column_name (str): The name of the column to check.
        placeholder_codes (list): List of codes considered 'missing'.
        threshold (float): Max % allowed before replacing instead of dropping.
        
    Returns:
        DataFrame: Cleaned DataFrame
    """
    df = df.copy()
    mask = df[column_name].isin(placeholder_codes)
    proportion = mask.mean()

    if proportion <= threshold and proportion != 0:
        print(f"Dropping {mask.sum()} rows from '{column_name}' with placeholder codes ({proportion * 100:.2f}%).")
        return df[~mask]
    elif proportion == 0:
        print(f"No rows in '{column_name}' contain placeholder codes.")
        return df
    else:
        print(f"Keeping all rows in '{column_name}' — {proportion * 100:.2f}% have placeholder codes (over {threshold*100:.1f}%).")
        df.loc[mask, column_name] = np.nan
        return df
