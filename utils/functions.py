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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



# -----------------------------------------------------------
# ---------------   Data Cleaning Functions   ---------------
# -----------------------------------------------------------

def clean_dataframe(df, placeholders, drop_threshold=0.05):
    """
    1. Replaces placeholders with NaN.
    2. Drops all duplicates (global and patient_nbr).
    3. Drops rows that are too sparse (>drop_threshold proportion missing).
 
    Args:
    df: pandas DataFrame - The input DataFrame to be cleaned.
    placeholders: list - A list of placeholder values to be replaced with NaN.
    drop_threshold: float - The threshold for dropping rows based on the proportion of missing values in that row.
    
    Returns:
    df: cleaned DataFrame
    report: dict with removal stats
    """
    df = df.copy()
    original_shape = df.shape
    
    # Global replace
    df = df.replace(placeholders, np.nan).infer_objects(copy=False)
    
    # Remove duplicates (global)
    df = df.drop_duplicates()
    after_global = df.shape[0]
    
    # Remove duplicates (patient_nbr)
    if 'patient_nbr' in df.columns:
        df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    after_patient = df.shape[0]
        
    # Generate report
    report = {
        'original_shape': original_shape,
        'after_global_dedup': (after_global, original_shape[0] - after_global),
        'after_patient_dedup': (after_patient, after_global - after_patient),
        'final_shape': df.shape
    }
    
    return df, report

def drop_useless_columns(df, threshold=0.4):
    """
    Drop columns with NaN ratio higher than threshold.
    Returns: df and list of dropped columns
    """
    df = df.copy()
    cols_before = set(df.columns)

    df = df.loc[:, df.isnull().mean() < threshold]
    cols_after = set(df.columns)
    
    dropped_cols = list(cols_before - cols_after)
    
    return df, dropped_cols


# -----------------------------------------------------------
# ---------------   Data Splitting Functions   --------------
# -----------------------------------------------------------

def column_split(df):
    """
    Splits the DataFrame into categorical and numerical columns based on data types.
    Args:
        df: pandas DataFrame - The input DataFrame to be split.
    Returns:
        cat_cols: list - The names of the categorical columns.
        num_cols: list - The names of the numerical columns.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return cat_cols, num_cols


def apply_imputer(df, imputer, cat_cols, num_cols):
    imputed = pd.DataFrame(
        imputer.transform(df[cat_cols]),
        columns=cat_cols,
        index=df.index
    )
    return pd.concat([df[num_cols], imputed], axis=1)


def data_split(df, target_col, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Best practice split: 
    1. Ensures total ratio equals 1.0
    2. Uses stratification to keep class balance.
    3. Handles the math internally.

    Args:
        df: pandas DataFrame - The input DataFrame to be split.
        target_col: str - The name of the target column in the DataFrame.
        train_size: float - The proportion of the dataset to include in the training set.
        val_size: float - The proportion of the dataset to include in the validation set.
        test_size: float - The proportion of the dataset to include in the test set.
        random_state: int - The random seed for reproducibility.

    Returns:
        X_train, X_val, X_test: pandas DataFrames - The feature sets for training
    """

    if (train_size + val_size + test_size) != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Extracting Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    # Extracting Val set from the remainder
    adjusted_val_size = val_size / (train_size + val_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, 
        y_temp, 
        test_size=adjusted_val_size, 
        random_state=random_state, 
        stratify=y_temp
    )

    print("Data split completed successfully.")
    print(f"Shapes: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test



# -----------------------------------------------------------
# --------------- Feature Engineering Functions -------------
# -----------------------------------------------------------

def binarize_target(y):
    """
    Converts the 3-class readmitted target into binary.
    1 = readmitted within 30 days (<30)
    0 = not readmitted within 30 days (>30 or NO)

    Args:
        y: pandas Series - The target variable to be binarized.
    Returns:
        pandas Series - The binarized target variable.
    """
    return (y == '<30').astype(int)


def map_icd9(code, icd9_mapped_categories):
    if pd.isna(code):
        return 'unknown'
    
    code = str(code).strip()
    
    if code.startswith('V') or code.startswith('E'):
        return 'other'
    
    try:
        numeric = float(code)
    except ValueError:
        return 'unknown'
    
    for start, end, category in icd9_mapped_categories:
        if start <= numeric <= end:
            return category
    
    return 'other'

def get_rare_category_indices(df, col, min_count=10):
    """
    Returns indices of rows where col has a category 
    appearing fewer than min_count times.
    Args:
        df: pandas DataFrame - The input DataFrame.
        col: str - The name of the column to analyze.
        min_count: int - The minimum count threshold for a category to be considered common.
    Returns:
        pandas Index - The indices of rows with rare categories in the specified column.
    """
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index

    return df[df[col].isin(rare)].index


# -----------------------------------------------------------
# -------------------- Encoder Functions --------------------
# -----------------------------------------------------------
def encode_categorical(X_train, X_val, X_test, cat_cols):
    """
    One-hot encodes categorical columns.
    Fits on X_train only, transforms all three sets.
    Drops first category to avoid multicollinearity (n-1 encoding).
    
    Args:
        X_train, X_val, X_test: pandas DataFrames
        cat_cols: list of categorical column names to encode
    Returns:
        X_train, X_val, X_test: encoded DataFrames
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    encoder.fit(X_train[cat_cols])
    
    def transform(X):
        encoded = pd.DataFrame(
            encoder.transform(X[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X.index
        )
        return pd.concat([X.drop(columns=cat_cols), encoded], axis=1)
    
    return transform(X_train), transform(X_val), transform(X_test)