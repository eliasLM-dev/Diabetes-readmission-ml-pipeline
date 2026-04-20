"""
Test suite for utils/functions.py

Covers:
- Data cleaning (clean_dataframe, drop_useless_columns)
- Data splitting (data_split)
- Target binarization (binarize_target)
- Column splitting (column_split)
- Imputation (apply_imputer)
- Feature engineering (map_icd9, get_rare_category_indices)
"""
import pandas as pd
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
import utils.functions as f


# -----------------------------------------------------------
# -----   Tests for clean_dataframe   -----
# -----------------------------------------------------------

def test_clean_dataframe_replaces_placeholders():
    """Placeholder values are replaced with NaN"""
    df = pd.DataFrame({
        'col1': [1, 2, '?'],
        'col2': [4, 'Unknown', 6]
    })
    cleaned_df, _ = f.clean_dataframe(df, ['?', 'Unknown'])

    assert pd.isna(cleaned_df.iloc[2, 0])
    assert pd.isna(cleaned_df.iloc[1, 1])


def test_clean_dataframe_removes_global_duplicates():
    """Exact duplicate rows are removed"""
    df = pd.DataFrame({
        'col1': [1, 1, 2],
        'col2': [4, 4, 5]
    })
    cleaned_df, report = f.clean_dataframe(df, [])

    assert cleaned_df.shape[0] == 2
    assert report['after_global_dedup'][1] == 1


def test_clean_dataframe_removes_patient_duplicates():
    """Duplicate patient_nbr rows are removed, keeping first"""
    df = pd.DataFrame({
        'patient_nbr': [1, 1, 2],
        'col1': [10, 20, 30]
    })
    cleaned_df, report = f.clean_dataframe(df, [])

    assert cleaned_df.shape[0] == 2
    assert cleaned_df[cleaned_df['patient_nbr'] == 1]['col1'].values[0] == 10
    assert report['after_patient_dedup'][1] == 1


def test_clean_dataframe_returns_report():
    """Report contains all expected keys"""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    _, report = f.clean_dataframe(df, [])

    assert 'original_shape' in report
    assert 'final_shape' in report
    assert 'after_global_dedup' in report
    assert 'after_patient_dedup' in report


def test_clean_dataframe_does_not_modify_original():
    """Original DataFrame is not modified"""
    df = pd.DataFrame({'col1': [1, '?', 3]})
    original_values = df['col1'].tolist()
    f.clean_dataframe(df, ['?'])

    assert df['col1'].tolist() == original_values


# -----------------------------------------------------------
# -----   Tests for drop_useless_columns   -----
# -----------------------------------------------------------

def test_drop_useless_columns_removes_sparse():
    """Columns exceeding NaN threshold are dropped"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [np.nan, np.nan, np.nan, 5],  # 75% NaN
        'col3': [1, 1, 1, 1]
    })
    cleaned_df, dropped = f.drop_useless_columns(df, threshold=0.4)

    assert 'col2' in dropped
    assert cleaned_df.shape[1] == 2


def test_drop_useless_columns_keeps_good_columns():
    """Columns below NaN threshold are retained"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [1, np.nan, 3, 4]  # 25% NaN
    })
    cleaned_df, dropped = f.drop_useless_columns(df, threshold=0.4)

    assert 'col2' not in dropped
    assert 'col2' in cleaned_df.columns


def test_drop_useless_columns_does_not_modify_original():
    """Original DataFrame is not modified"""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [np.nan, np.nan, np.nan]
    })
    original_cols = df.columns.tolist()
    f.drop_useless_columns(df, threshold=0.4)

    assert df.columns.tolist() == original_cols


# -----------------------------------------------------------
# -----   Tests for data_split   -----
# -----------------------------------------------------------

def test_data_split_correct_proportions():
    """Split produces correct train/val/test proportions"""
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'target': np.random.choice([0, 1], 1000)
    })
    X_train, X_val, X_test, y_train, y_val, y_test = f.data_split(
        df, 'target', train_size=0.6, val_size=0.2, test_size=0.2
    )
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]

    assert abs(X_train.shape[0] / total - 0.6) < 0.02
    assert abs(X_val.shape[0] / total - 0.2) < 0.02
    assert abs(X_test.shape[0] / total - 0.2) < 0.02
    assert total == 1000


def test_data_split_invalid_ratios():
    """ValueError is raised when ratios do not sum to 1.0"""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [0, 1, 0]
    })
    with pytest.raises(ValueError):
        f.data_split(df, 'target', train_size=0.5, val_size=0.3, test_size=0.3)


def test_data_split_no_index_overlap():
    """Train, val, and test sets share no indices"""
    df = pd.DataFrame({
        'feature1': range(100),
        'target': [0, 1] * 50
    })
    X_train, X_val, X_test, _, _, _ = f.data_split(df, 'target')

    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert len(train_idx & val_idx) == 0
    assert len(train_idx & test_idx) == 0
    assert len(val_idx & test_idx) == 0


def test_data_split_xy_index_alignment():
    """X and y sets share identical indices after split"""
    df = pd.DataFrame({
        'feature1': range(100),
        'target': [0, 1] * 50
    })
    X_train, X_val, X_test, y_train, y_val, y_test = f.data_split(df, 'target')

    assert list(X_train.index) == list(y_train.index)
    assert list(X_val.index) == list(y_val.index)
    assert list(X_test.index) == list(y_test.index)


def test_data_split_stratification():
    """Class balance is preserved across all splits"""
    df = pd.DataFrame({
        'feature1': range(1000),
        'target': [0] * 900 + [1] * 100
    })
    _, _, _, y_train, y_val, y_test = f.data_split(df, 'target')

    for y in [y_train, y_val, y_test]:
        ratio = y.mean()
        assert abs(ratio - 0.1) < 0.02


# -----------------------------------------------------------
# -----   Tests for binarize_target   -----
# -----------------------------------------------------------

def test_binarize_target_correct_mapping():
    """<30 maps to 1, all other values map to 0"""
    y = pd.Series(['<30', '>30', 'NO', '<30'])
    result = f.binarize_target(y)

    assert list(result) == [1, 0, 0, 1]


def test_binarize_target_no_nulls():
    """Binarized target contains no NaN values"""
    y = pd.Series(['<30', '>30', 'NO'])
    result = f.binarize_target(y)

    assert result.isnull().sum() == 0


def test_binarize_target_returns_binary():
    """Output contains only 0 and 1"""
    y = pd.Series(['<30', '>30', 'NO', '<30', 'NO'])
    result = f.binarize_target(y)

    assert set(result.unique()).issubset({0, 1})


# -----------------------------------------------------------
# -----   Tests for column_split   -----
# -----------------------------------------------------------

def test_column_split_separates_correctly():
    """Numerical and categorical columns are correctly identified"""
    df = pd.DataFrame({
        'num_col': [1, 2, 3],
        'cat_col': ['a', 'b', 'c']
    })
    cat_cols, num_cols = f.column_split(df)

    assert 'cat_col' in cat_cols
    assert 'num_col' in num_cols


def test_column_split_no_overlap():
    """No column appears in both categorical and numerical lists"""
    df = pd.DataFrame({
        'num_col': [1, 2, 3],
        'cat_col': ['a', 'b', 'c']
    })
    cat_cols, num_cols = f.column_split(df)

    assert len(set(cat_cols) & set(num_cols)) == 0


def test_column_split_all_columns_accounted():
    """Every column in the DataFrame appears in exactly one list"""
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [1.0, 2.0, 3.0],
        'cat1': ['a', 'b', 'c']
    })
    cat_cols, num_cols = f.column_split(df)

    assert len(cat_cols) + len(num_cols) == len(df.columns)


# -----------------------------------------------------------
# -----   Tests for apply_imputer   -----
# -----------------------------------------------------------

def test_apply_imputer_removes_nulls():
    """No NaN values remain in categorical columns after imputation"""
    X_train = pd.DataFrame({
        'cat': ['a', 'b', 'a', 'a'],
        'num': [1, 2, 3, 4]
    })
    X_test = pd.DataFrame({
        'cat': ['a', None, 'b', 'a'],
        'num': [1, 2, 3, 4]
    })
    cat_cols, num_cols = ['cat'], ['num']

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_train[cat_cols])

    result = f.apply_imputer(X_test, imputer, cat_cols, num_cols)

    assert result['cat'].isnull().sum() == 0


def test_apply_imputer_preserves_numerical_columns():
    """Numerical columns are untouched by categorical imputer"""
    X_train = pd.DataFrame({
        'cat': ['a', 'b', 'a', 'a'],
        'num': [1, 2, 3, 4]
    })
    X_test = pd.DataFrame({
        'cat': [None, 'b', 'a', 'a'],
        'num': [10, 20, 30, 40]
    })
    cat_cols, num_cols = ['cat'], ['num']

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_train[cat_cols])

    result = f.apply_imputer(X_test, imputer, cat_cols, num_cols)

    assert list(result['num']) == [10, 20, 30, 40]


def test_apply_imputer_preserves_index():
    """DataFrame index is preserved after imputation"""
    X_train = pd.DataFrame(
        {'cat': ['a', 'b', 'a'], 'num': [1, 2, 3]},
        index=[10, 20, 30]
    )
    cat_cols, num_cols = ['cat'], ['num']

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_train[cat_cols])

    result = f.apply_imputer(X_train, imputer, cat_cols, num_cols)

    assert list(result.index) == [10, 20, 30]


# -----------------------------------------------------------
# -----   Tests for map_icd9   -----
# -----------------------------------------------------------

ICD9_CATEGORIES = [
    (250, 250.99, "diabetes"),
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
]


def test_map_icd9_known_code():
    """Known ICD-9 codes map to correct category"""
    assert f.map_icd9('250', ICD9_CATEGORIES) == 'diabetes'
    assert f.map_icd9('410', ICD9_CATEGORIES) == 'circulatory'
    assert f.map_icd9('486', ICD9_CATEGORIES) == 'respiratory'


def test_map_icd9_decimal_code():
    """Decimal ICD-9 codes map correctly"""
    assert f.map_icd9('250.8', ICD9_CATEGORIES) == 'diabetes'


def test_map_icd9_v_codes():
    """V codes map to other"""
    assert f.map_icd9('V71', ICD9_CATEGORIES) == 'other'
    assert f.map_icd9('V58', ICD9_CATEGORIES) == 'other'


def test_map_icd9_e_codes():
    """E codes map to other"""
    assert f.map_icd9('E123', ICD9_CATEGORIES) == 'other'


def test_map_icd9_nan():
    """NaN input maps to unknown"""
    assert f.map_icd9(np.nan, ICD9_CATEGORIES) == 'unknown'


def test_map_icd9_unmapped_valid_code():
    """Valid numeric code outside all ranges maps to other"""
    assert f.map_icd9('999', ICD9_CATEGORIES) == 'other'


def test_map_icd9_unparseable_code():
    """Non-numeric, non-V/E code maps to unknown"""
    assert f.map_icd9('XYZ', ICD9_CATEGORIES) == 'unknown'


# -----------------------------------------------------------
# -----   Tests for get_rare_category_indices   -----
# -----------------------------------------------------------

def test_get_rare_category_indices_finds_rare():
    """Rows with rare category values are correctly identified"""
    df = pd.DataFrame({'col': ['a'] * 100 + ['b'] * 5})
    idx = f.get_rare_category_indices(df, 'col', min_count=10)

    assert len(idx) == 5


def test_get_rare_category_indices_empty_when_none_rare():
    """No indices returned when all categories exceed threshold"""
    df = pd.DataFrame({'col': ['a'] * 100 + ['b'] * 100})
    idx = f.get_rare_category_indices(df, 'col', min_count=10)

    assert len(idx) == 0


def test_get_rare_category_indices_correct_rows_identified():
    """Correct row indices are returned for rare categories"""
    df = pd.DataFrame({'col': ['a', 'a', 'a', 'b']})
    idx = f.get_rare_category_indices(df, 'col', min_count=2)

    assert 3 in idx
    assert 0 not in idx