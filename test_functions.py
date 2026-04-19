"""
Simple tests for functions.py
Tests the core logic of data cleaning and splitting functions.
"""
import pandas as pd
import numpy as np
import pytest
from utils.functions import clean_dataframe, drop_useless_columns, data_split


# -----------------------------------------------------------
# -----   Tests for clean_dataframe   -----
# -----------------------------------------------------------

def test_clean_dataframe_replaces_placeholders():
    """Test that placeholders are replaced with NaN"""
    df = pd.DataFrame({
        'col1': [1, 2, '?'],
        'col2': [4, 'Unknown', 6]
    })
    
    placeholders = ['?', 'Unknown']
    cleaned_df, _ = clean_dataframe(df, placeholders)
    
    # Check that placeholders are now NaN
    assert pd.isna(cleaned_df.iloc[2, 0])  # '?' should be NaN
    assert pd.isna(cleaned_df.iloc[1, 1])  # 'Unknown' should be NaN


def test_clean_dataframe_removes_duplicates():
    """Test that global duplicates are removed"""
    df = pd.DataFrame({
        'col1': [1, 1, 2],
        'col2': [4, 4, 5]
    })
    
    cleaned_df, report = clean_dataframe(df, [])
    
    # Should have 2 rows after removing duplicate row
    assert cleaned_df.shape[0] == 2
    assert report['original_shape'][0] == 3


def test_clean_dataframe_returns_report():
    """Test that clean_dataframe returns a report dict"""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    _, report = clean_dataframe(df, [])
    
    # Check report has expected keys
    assert 'original_shape' in report
    assert 'final_shape' in report
    assert 'after_global_dedup' in report


# -----------------------------------------------------------
# -----   Tests for drop_useless_columns   -----
# -----------------------------------------------------------

def test_drop_useless_columns_removes_sparse():
    """Test that columns with high NaN ratio are dropped"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [np.nan, np.nan, np.nan, 5],  # 75% NaN
        'col3': [1, 1, 1, 1]
    })
    
    # With threshold=0.4, col2 (75% NaN) should be dropped
    cleaned_df, dropped = drop_useless_columns(df, threshold=0.4)
    
    assert 'col2' in dropped
    assert cleaned_df.shape[1] == 2  # Only col1 and col3 remain


def test_drop_useless_columns_keeps_good_columns():
    """Test that columns below threshold are kept"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [1, np.nan, 3, 4]  # 25% NaN - should be kept with threshold=0.4
    })
    
    cleaned_df, dropped = drop_useless_columns(df, threshold=0.4)
    
    assert 'col2' not in dropped
    assert 'col2' in cleaned_df.columns


# -----------------------------------------------------------
# -----   Tests for data_split   -----
# -----------------------------------------------------------

def test_data_split_correct_proportions():
    """Test that data is split in correct proportions"""
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(
        df, 'target', train_size=0.6, val_size=0.2, test_size=0.2
    )
    
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    
    # Check approximate proportions (allow small variance due to stratification)
    assert 550 < X_train.shape[0] < 650  # ~60%
    assert 150 < X_val.shape[0] < 250    # ~20%
    assert 150 < X_test.shape[0] < 250   # ~20%
    assert total == 1000


def test_data_split_invalid_ratios():
    """Test that data_split raises error if ratios don't sum to 1.0"""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [0, 1, 0]
    })
    
    with pytest.raises(ValueError):
        data_split(df, 'target', train_size=0.5, val_size=0.3, test_size=0.3)


def test_data_split_no_data_leakage():
    """Test that train, val, test sets don't overlap"""
    df = pd.DataFrame({
        'feature1': range(100),
        'target': [0, 1] * 50
    })
    
    X_train, X_val, X_test, _, _, _ = data_split(df, 'target')
    
    # Create index sets and verify no overlap
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0


# -----------------------------------------------------------
# -----   Simple sanity check   -----
# -----------------------------------------------------------

def test_imports_work():
    """Sanity check - can we import without errors"""
    assert True  # If imports fail, test won't run