import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data_sample(file_path, sample_size=50000):
    '''
    Load a sample of the data for initial exploration
    '''
    print(f"Loading {sample_size} rows from {file_path}...")
    
    # Read sample data
    df = pd.read_csv(file_path, nrows=sample_size)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    return df

def basic_info(df):
    '''
    Display basic information about the dataset
    '''
    print("\n=== BASIC INFORMATION ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_percent
    })
    print(missing_df[missing_df['missing_count'] > 0])
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())

def analyze_target(df):
    '''
    Analyze the target variable (loadWeight)
    '''
    print("\n=== TARGET VARIABLE ANALYSIS (loadWeight) ===")
    
    # Clean the target variable
    df['loadWeight_clean'] = pd.to_numeric(df['loadWeight'], errors='coerce')
    
    # Basic stats
    print(f"Count: {df['loadWeight_clean'].count()}")
    print(f"Mean: {df['loadWeight_clean'].mean():.2f}")
    print(f"Std: {df['loadWeight_clean'].std():.2f}")
    print(f"Min: {df['loadWeight_clean'].min():.2f}")
    print(f"25%: {df['loadWeight_clean'].quantile(0.25):.2f}")
    print(f"50% (Median): {df['loadWeight_clean'].quantile(0.50):.2f}")
    print(f"75%: {df['loadWeight_clean'].quantile(0.75):.2f}")
    print(f"Max: {df['loadWeight_clean'].max():.2f}")
    
    # Count missing/invalid
    missing_count = df['loadWeight_clean'].isnull().sum()
    print(f"Missing/Invalid values: {missing_count} ({missing_count/len(df)*100:.2f}%)")
    
    # Identify outliers using IQR method
    Q1 = df['loadWeight_clean'].quantile(0.25)
    Q3 = df['loadWeight_clean'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['loadWeight_clean'] < lower_bound) | (df['loadWeight_clean'] > upper_bound)]
    print(f"Potential outliers (IQR method): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    # Reasonable range for waste weight (based on domain knowledge)
    reasonable_min, reasonable_max = 0, 50000  # lbs
    unreasonable = df[(df['loadWeight_clean'] < reasonable_min) | (df['loadWeight_clean'] > reasonable_max)]
    print(f"Unreasonable values (<0 or >50000 lbs): {len(unreasonable)} ({len(unreasonable)/len(df)*100:.2f}%)")

def analyze_categorical(df):
    '''
    Analyze categorical variables
    '''
    print("\n=== CATEGORICAL VARIABLES ANALYSIS ===")
    
    categorical_cols = ['loadType', 'dropoffSite', 'routeType', 'routeNumber']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Missing values: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)")
            
            # Top 10 most frequent values
            top_values = df[col].value_counts().head(10)
            print("  Top 10 values:")
            for val, count in top_values.items():
                print(f"    {val}: {count} ({count/len(df)*100:.2f}%)")

def analyze_temporal(df):
    '''
    Analyze temporal variables
    '''
    print("\n=== TEMPORAL VARIABLES ANALYSIS ===")
    
    # Convert date columns
    df['reportDate_parsed'] = pd.to_datetime(df['reportDate'], errors='coerce')
    df['loadTime_parsed'] = pd.to_datetime(df['loadTime'], errors='coerce')
    
    print("reportDate:")
    print(f"  Range: {df['reportDate_parsed'].min()} to {df['reportDate_parsed'].max()}")
    print(f"  Missing: {df['reportDate_parsed'].isnull().sum()} ({df['reportDate_parsed'].isnull().sum()/len(df)*100:.2f}%)")
    
    print("loadTime:")
    print(f"  Range: {df['loadTime_parsed'].min()} to {df['loadTime_parsed'].max()}")
    print(f"  Missing: {df['loadTime_parsed'].isnull().sum()} ({df['loadTime_parsed'].isnull().sum()/len(df)*100:.2f}%)")

def main():
    '''
    Main exploration function
    '''
    print("Starting data exploration...")
    
    # Load data
    df = load_data_sample('largeClean.csv', sample_size=50000)
    
    # Basic information
    basic_info(df)
    
    # Target variable analysis
    analyze_target(df)
    
    # Categorical variables
    analyze_categorical(df)
    
    # Temporal variables
    analyze_temporal(df)
    
    print("\n=== EXPLORATION COMPLETE ===")

if __name__ == '__main__':
    main()
