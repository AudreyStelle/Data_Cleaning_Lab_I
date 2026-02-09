#%%
# %%
# Load the data
college_completion = pd.read_csv('college_completion.csv')
print(f"Loaded data: {college_completion.shape}")
college_completion.head()
#%%
# %%
# %%
# Test just the imports and first few functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("Imports successful!")
#%%

# %%
"""
Data Pipeline Functions for ML Preprocessing
Step 4: Modular pipeline functions for data cleaning and preparation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# ========================================
# GENERAL UTILITY FUNCTIONS
# ========================================

def load_data(filepath_or_url):
    """
    Load data from a CSV file or URL.
    
    Parameters:
    -----------
    filepath_or_url : str
        Path to local file or URL to CSV data
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    return pd.read_csv(filepath_or_url)


def convert_to_categorical(df, column_list):
    """
    Convert specified columns to categorical data type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_list : list
        List of column names to convert to categorical
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with specified columns converted to categorical
    """
    df_copy = df.copy()
    df_copy[column_list] = df_copy[column_list].astype("category")
    return df_copy


def scale_numeric_columns(df, columns, method='minmax'):
    """
    Scale numeric columns using MinMaxScaler or StandardScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to scale
    method : str, default='minmax'
        Scaling method: 'minmax' or 'standard'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with scaled columns
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("method must be 'minmax' or 'standard'")
    
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy


def one_hot_encode(df, category_columns, drop_first=True):
    """
    One-hot encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    category_columns : list
        List of categorical column names to encode
    drop_first : bool, default=True
        Whether to drop the first category to avoid multicollinearity
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one-hot encoded columns
    """
    return pd.get_dummies(df, columns=category_columns, drop_first=drop_first)


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns_to_drop : list
        List of column names to drop
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns removed
    """
    return df.drop(columns=columns_to_drop, errors='ignore')


def create_train_test_split(df, target_column, test_size=0.3, random_state=42, stratify=True):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target variable column
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        Whether to use stratified sampling
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    stratify_var = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_var
    )
    
    return X_train, X_test, y_train, y_test

# %%
# ========================================
# COLLEGE COMPLETION SPECIFIC FUNCTIONS
# ========================================

def create_binned_column(df, column_name, bins, labels, new_column_name=None):
    """
    Create a binned categorical variable from a continuous variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_name : str
        Name of continuous column to bin
    bins : list
        Bin edges for pd.cut
    labels : list
        Labels for each bin
    new_column_name : str, optional
        Name for new column. If None, uses column_name + '_group'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with new binned column
    """
    df_copy = df.copy()
    
    if new_column_name is None:
        new_column_name = f"{column_name}_group"
    
    df_copy[new_column_name] = pd.cut(
        df_copy[column_name],
        bins=bins,
        labels=labels
    )
    
    return df_copy


def create_binary_target_from_quantile(df, column_name, quantile=0.75, new_column_name=None):
    """
    Create a binary target variable based on quantile threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_name : str
        Name of continuous column
    quantile : float, default=0.75
        Quantile threshold (e.g., 0.75 for top quartile)
    new_column_name : str, optional
        Name for new binary column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with new binary target column
    """
    df_copy = df.copy()
    
    if new_column_name is None:
        new_column_name = f"{column_name}_high"
    
    threshold = df_copy[column_name].quantile(quantile)
    df_copy[new_column_name] = (df_copy[column_name] >= threshold).astype(int)
    
    return df_copy


def prep_college_completion_data(filepath_or_url):
    """
    Complete preprocessing pipeline for College Completion dataset.
    
    Parameters:
    -----------
    filepath_or_url : str
        Path to college completion CSV file or URL
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe ready for train-test split
    """
    df = load_data(filepath_or_url)
    
    categorical_cols = ['level', 'control', 'hbcu', 'flagship']
    df = convert_to_categorical(df, categorical_cols)
    
    df = create_binned_column(
        df,
        column_name='pell_value',
        bins=[0, 33, 66, 100],
        labels=['Low', 'Medium', 'High'],
        new_column_name='pell_group'
    )
    
    df = create_binary_target_from_quantile(
        df,
        column_name='grad_150_value',
        quantile=0.75,
        new_column_name='high_completion_q'
    )
    
    vsa_cols = [col for col in df.columns if col.startswith('vsa_')]
    other_cols_to_drop = [
        'exp_award_value', 'exp_award_state_value', 'exp_award_natl_value',
        'exp_award_percentile', 'fte_value', 'fte_percentile', 
        'med_sat_value', 'med_sat_percentile', 'endow_value', 'endow_percentile'
    ]
    all_cols_to_drop = vsa_cols + other_cols_to_drop
    df = drop_columns(df, all_cols_to_drop)
    
    category_list = ['level', 'control', 'hbcu', 'flagship', 'pell_group']
    df = one_hot_encode(df, category_list, drop_first=True)
    
    return df


def get_college_completion_train_test(filepath_or_url, train_size=0.55, random_state=42):
    """
    Generate train and test datasets for College Completion problem.
    
    Parameters:
    -----------
    filepath_or_url : str
        Path to college completion CSV file or URL
    train_size : float, default=0.55
        Proportion of data to use for training
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    df = prep_college_completion_data(filepath_or_url)
    
    test_size = 1 - train_size
    X_train, X_test, y_train, y_test = create_train_test_split(
        df,
        target_column='high_completion_q',
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )
    
    return X_train, X_test, y_train, y_test

# %%
# ========================================
# JOB PLACEMENT SPECIFIC FUNCTIONS
# ========================================

def handle_missing_salary(df):
    """
    Handle missing salary values by filtering to only placed students.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'status' and 'salary' columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame filtered to only placed students (removes missing salaries)
    """
    return df[df['status'] == 'Placed'].copy()


def create_binary_target_from_categorical(df, column_name, positive_value, new_column_name=None):
    """
    Create a binary target from a categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_name : str
        Name of categorical column
    positive_value : str
        Value to encode as 1 (positive class)
    new_column_name : str, optional
        Name for new binary column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with new binary column
    """
    df_copy = df.copy()
    
    if new_column_name is None:
        new_column_name = f"{column_name}_bool"
    
    df_copy[new_column_name] = (df_copy[column_name] == positive_value).astype(int)
    
    return df_copy


def scale_all_numeric_columns(df, method='minmax', exclude_columns=None):
    """
    Scale all numeric columns in a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str, default='minmax'
        Scaling method: 'minmax' or 'standard'
    exclude_columns : list, optional
        List of numeric columns to exclude from scaling
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all numeric columns scaled
    """
    df_copy = df.copy()
    
    numeric_cols = list(df_copy.select_dtypes(include='number').columns)
    
    if exclude_columns:
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    
    if numeric_cols:
        df_copy = scale_numeric_columns(df_copy, numeric_cols, method=method)
    
    return df_copy


def prep_job_placement_data(filepath_or_url, filter_placed=False):
    """
    Complete preprocessing pipeline for Job Placement dataset.
    
    Parameters:
    -----------
    filepath_or_url : str
        Path to job placement CSV file or URL
    filter_placed : bool, default=False
        Whether to filter to only placed students (removes missing salaries)
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe ready for train-test split
    """
    df = load_data(filepath_or_url)
    
    categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
    df = convert_to_categorical(df, categorical_cols)
    
    df = create_binary_target_from_categorical(
        df,
        column_name='status',
        positive_value='Placed',
        new_column_name='placed_bool'
    )
    
    mba_cutoff = df['mba_p'].quantile(0.75)
    df['mba_high'] = pd.cut(
        df['mba_p'],
        bins=[-1, mba_cutoff, 100],
        labels=[0, 1]
    )
    
    if filter_placed:
        df = handle_missing_salary(df)
    
    df = scale_all_numeric_columns(df, method='minmax', exclude_columns=['placed_bool'])
    
    category_list = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'mba_high']
    df = one_hot_encode(df, category_list, drop_first=True)
    
    df = drop_columns(df, ['status'])
    
    return df


def get_job_placement_train_test(filepath_or_url, test_size=0.3, random_state=42, filter_placed=False):
    """
    Generate train and test datasets for Job Placement problem.
    
    Parameters:
    -----------
    filepath_or_url : str
        Path to job placement CSV file or URL
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    filter_placed : bool, default=False
        Whether to filter to only placed students
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    df = prep_job_placement_data(filepath_or_url, filter_placed=filter_placed)
    
    X_train, X_test, y_train, y_test = create_train_test_split(
        df,
        target_column='placed_bool',
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )
    
    return X_train, X_test, y_train, y_test

# %%
# ========================================
# TEST THE PIPELINES
# ========================================

# Test College Completion Pipeline
print("=" * 60)
print("COLLEGE COMPLETION PIPELINE")
print("=" * 60)

X_train_cc, X_test_cc, y_train_cc, y_test_cc = get_college_completion_train_test(
    'college_completion.csv',
    train_size=0.55,
    random_state=42
)

print(f"\nTrain set shape: {X_train_cc.shape}")
print(f"Test set shape: {X_test_cc.shape}")
print(f"Target distribution in train: {y_train_cc.value_counts().to_dict()}")
print(f"Target distribution in test: {y_test_cc.value_counts().to_dict()}")

# %%
# Test Job Placement Pipeline
print("\n" + "=" * 60)
print("JOB PLACEMENT PIPELINE")
print("=" * 60)

X_train_jp, X_test_jp, y_train_jp, y_test_jp = get_job_placement_train_test(
    'Placement_Data_Full_Class.csv',
    test_size=0.3,
    random_state=42,
    filter_placed=False
)

print(f"\nTrain set shape: {X_train_jp.shape}")
print(f"Test set shape: {X_test_jp.shape}")
print(f"Target distribution in train: {y_train_jp.value_counts().to_dict()}")
print(f"Target distribution in test: {y_test_jp.value_counts().to_dict()}")

print("\n" + "=" * 60)
print("✓ ALL PIPELINES WORKING!")
print("=" * 60)
#%%

# %%
