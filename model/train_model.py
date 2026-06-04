import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_and_preprocess_data(file_path, sample_size=100000):
    '''
    Load and preprocess the waste data for modeling
    '''
    print('Loading data...')
    
    # Read data in chunks to handle large file
    chunks = []
    rows_read = 0
    
    for chunk in pd.read_csv(file_path, chunksize=10000):
        chunks.append(chunk)
        rows_read += len(chunk)
        if rows_read >= sample_size:
            break
    
    df = pd.concat(chunks, ignore_index=True)
    print(f'Loaded {len(df)} rows for processing')
    
    # Clean loadWeight column - remove invalid values
    df['loadWeight'] = pd.to_numeric(df['loadWeight'], errors='coerce')
    
    # Remove rows with missing or extreme weights (based on domain knowledge)
    # Keep weights between 0 and 50000 lbs (reasonable range for waste loads)
    df = df[(df['loadWeight'] >= 0) & (df['loadWeight'] <= 50000)]
    print(f'After cleaning weight ranges: {len(df)} rows')
    
    # Convert date columns
    df['reportDate'] = pd.to_datetime(df['reportDate'], errors='coerce')
    df['loadTime'] = pd.to_datetime(df['loadTime'], errors='coerce')
    
    # Extract features from dates
    df['reportYear'] = df['reportDate'].dt.year
    df['reportMonth'] = df['reportDate'].dt.month
    df['reportDayOfWeek'] = df['reportDate'].dt.dayofweek
    df['loadHour'] = df['loadTime'].dt.hour
    
    # Handle missing values in categorical columns
    categorical_cols = ['loadType', 'dropoffSite', 'routeType', 'routeNumber']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Load preprocessor to get consistent encoding
    from utils.preprocessor import WasteDataPreprocessor
    preprocessor = WasteDataPreprocessor.load('preprocessing/waste_preprocessor.pkl')
    
    # Transform data using preprocessor
    X, y = preprocessor.transform(df)
    
    print(f'Features: {list(X.columns)}')
    print(f'Target variable range: {y.min():.2f} to {y.max():.2f}')
    print(f'Target mean: {y.mean():.2f}')
    
    return X, y, preprocessor

def train_model(X, y):
    '''
    Train a Random Forest Regressor model
    '''
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print('Training Random Forest model...')
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate model
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f'Training MAE: {train_mae:.2f}')
    print(f'Test MAE: {test_mae:.2f}')
    print(f'Training R²: {train_r2:.4f}')
    print(f'Test R²: {test_r2:.4f}')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print('')
    print('Feature Importance:')
    print(feature_importance)
    
    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

def save_model(model, preprocessor, metrics, model_path='model/waste_predictor.pkl'):
    '''
    Save the trained model and preprocessing objects
    '''
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'metrics': metrics
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f'Model saved to {model_path}')

def main():
    '''
    Main function to run the training pipeline
    '''
    print('Starting waste prediction model training...')
    
    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess_data(
        'largeClean.csv', 
        sample_size=100000  # Use 100k samples for faster training
    )
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Save model
    save_model(model, preprocessor, metrics)
    
    print('Training completed successfully!')

if __name__ == '__main__':
    main()
