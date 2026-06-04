import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import sys
# Add the parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import WasteDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

def create_sample_preprocessor(file_path='largeClean.csv', sample_size=10000):
    '''
    Create and fit a preprocessor on a sample of the data
    '''
    print(f'Loading {sample_size} rows to create preprocessor...')
    
    # Load sample data
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Create and fit preprocessor
    preprocessor = WasteDataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    print(f'Preprocessed data shape: X={X.shape}, y={y.shape}')
    print(f'Feature names: {list(X.columns)}')
    
    return preprocessor

def main():
    '''
    Main function to demonstrate preprocessing
    '''
    print('Creating waste data preprocessor...')
    
    # Create preprocessor from sample data
    preprocessor = create_sample_preprocessor('largeClean.csv', sample_size=10000)
    
    # Save preprocessor
    os.makedirs('preprocessing', exist_ok=True)
    preprocessor.save('preprocessing/waste_preprocessor.pkl')
    
    # Example usage
    print('')
    print('=== Example Usage ===')
    # Load new data and transform
    new_data = pd.read_csv('largeClean.csv', nrows=1000)
    X_new, y_new = preprocessor.transform(new_data)
    print(f'New data shape: X={X_new.shape}, y={y_new.shape}')

if __name__ == '__main__':
    main()
