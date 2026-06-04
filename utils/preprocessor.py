import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class WasteDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, df):
        '''
        Fit the preprocessor on the training data
        '''
        print('Fitting preprocessor...')
        
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Clean loadWeight column - convert to numeric, handle errors
        data['loadWeight'] = pd.to_numeric(data['loadWeight'], errors='coerce')
        
        # Filter to reasonable weight range (0-50000 lbs based on exploration)
        data = data[(data['loadWeight'] >= 0) & (data['loadWeight'] <= 50000)]
        
        # Convert date columns
        data['reportDate'] = pd.to_datetime(data['reportDate'], errors='coerce')
        data['loadTime'] = pd.to_datetime(data['loadTime'], errors='coerce')
        
        # Extract temporal features
        data['reportYear'] = data['reportDate'].dt.year.fillna(0).astype(int)
        data['reportMonth'] = data['reportDate'].dt.month.fillna(0).astype(int)
        data['reportDayOfWeek'] = data['reportDate'].dt.dayofweek.fillna(0).astype(int)
        data['loadHour'] = data['loadTime'].dt.hour.fillna(0).astype(int)
        
        # Handle missing values in categorical columns
        categorical_cols = ['loadType', 'dropoffSite', 'routeType', 'routeNumber']
        for col in categorical_cols:
            data[col] = data[col].fillna('Unknown')
        
        # Fit label encoders for categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            data[col + '_encoded'] = le.fit_transform(data[col])
            self.label_encoders[col] = le
        
        # Define feature columns for modeling
        self.feature_columns = [
            'reportYear', 'reportMonth', 'reportDayOfWeek', 'loadHour',
            'loadType_encoded', 'dropoffSite_encoded', 'routeType_encoded', 'routeNumber_encoded'
        ]
        
        self.is_fitted = True
        print(f'Preprocessor fitted with {len(self.feature_columns)} features')
        print(f'Feature columns: {self.feature_columns}')
        
        return self
    
    def transform(self, df):
        '''
        Transform data using the fitted preprocessor
        '''
        if not self.is_fitted:
            raise ValueError('Preprocessor must be fitted before transform')
        
        print('Transforming data...')
        
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Store original loadWeight if present (for training/validation)
        has_load_weight = 'loadWeight' in data.columns
        original_load_weight = data['loadWeight'].copy() if has_load_weight else None
        
        # Clean loadWeight column - convert to numeric, handle errors
        if has_load_weight:
            data['loadWeight'] = pd.to_numeric(data['loadWeight'], errors='coerce')
            
            # Filter to reasonable weight range (0-50000 lbs based on exploration)
            data = data[(data['loadWeight'] >= 0) & (data['loadWeight'] <= 50000)]
        else:
            # For prediction, create a dummy loadWeight column to avoid errors
            data['loadWeight'] = 0.0  # dummy value
        
        # Handle date/time features: use individual components if available, otherwise parse strings
        # Check if we have the individual components for the temporal features
        if all(col in data.columns for col in ['reportYear', 'reportMonth', 'reportDayOfWeek', 'loadHour']) and \
           not data[['reportYear', 'reportMonth', 'reportDayOfWeek', 'loadHour']].isnull().all().all():
            # Use the individual components directly
            data['reportYear'] = data['reportYear'].fillna(0).astype(int)
            data['reportMonth'] = data['reportMonth'].fillna(0).astype(int)
            data['reportDayOfWeek'] = data['reportDayOfWeek'].fillna(0).astype(int)
            data['loadHour'] = data['loadHour'].fillna(0).astype(int)
        else:
            # Check if we have the date string columns
            if 'reportDate' in data.columns and 'loadTime' in data.columns:
                # Fall back to parsing the date strings
                data['reportDate'] = pd.to_datetime(data['reportDate'], errors='coerce')
                data['loadTime'] = pd.to_datetime(data['loadTime'], errors='coerce')
                
                # Extract features from dates
                data['reportYear'] = data['reportDate'].dt.year.fillna(0).astype(int)
                data['reportMonth'] = data['reportDate'].dt.month.fillna(0).astype(int)
                data['reportDayOfWeek'] = data['reportDate'].dt.dayofweek.fillna(0).astype(int)
                data['loadHour'] = data['loadTime'].dt.hour.fillna(0).astype(int)
            else:
                # If we don't have either, set default values (should not happen in normal use)
                data['reportYear'] = 0
                data['reportMonth'] = 0
                data['reportDayOfWeek'] = 0
                data['loadHour'] = 0
        
        # Handle missing values in categorical columns
        categorical_cols = ['loadType', 'dropoffSite', 'routeType', 'routeNumber']
        for col in categorical_cols:
            if col not in data.columns:
                data[col] = 'Unknown'
            else:
                data[col] = data[col].fillna('Unknown')
        
        # Transform categorical variables using fitted label encoders
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unseen labels by setting them to a known class
                le = self.label_encoders[col]
                # Map unknown values to the first class (index 0)
                data[col + '_encoded'] = data[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                # If encoder doesn't exist, create a default one
                le = LabelEncoder()
                le.fit(data[col].unique())
                data[col + '_encoded'] = le.transform(data[col])
                self.label_encoders[col] = le
        
        # Select and return features
        X = data[self.feature_columns].fillna(0)
        
        # Return features and target if available (for training/validation)
        if has_load_weight and original_load_weight is not None:
            # Use the original loadWeight values for y (before filtering)
            y = original_load_weight.reindex(X.index).fillna(original_load_weight.median())
            return X, y
        else:
            # For prediction, return only X
            return X
    
    def fit_transform(self, df):
        '''
        Fit the preprocessor and transform the data
        '''
        return self.fit(df).transform(df)
    
    def save(self, filepath):
        '''
        Save the preprocessor to disk
        '''
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f'Preprocessor saved to {filepath}')
    
    @classmethod
    def load(cls, filepath):
        '''
        Load a preprocessor from disk
        '''
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f'Preprocessor loaded from {filepath}')
        return preprocessor
