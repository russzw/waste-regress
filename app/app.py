from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import WasteDataPreprocessor

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_loaded = False

def load_model():
    '''Load the trained model and preprocessor'''
    global model, preprocessor, model_loaded
    
    try:
        # Construct path relative to this file
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'waste_predictor.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            model_loaded = True
            print('Model loaded successfully!')
            return True
        else:
            print(f'Model file not found at {model_path}. Please train the model first.')
            return False
    except Exception as e:
        print(f'Error loading model: {e}')
        return False

@app.route('/')
def index():
    '''Main dashboard page'''
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''Handle prediction requests'''
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        data = request.get_json()
        
        # Create DataFrame from input
        input_data = pd.DataFrame([data])
        
        # Preprocess the input (returns X only for prediction)
        result = preprocessor.transform(input_data)
        if isinstance(result, tuple):
            X = result[0]
        else:
            X = result
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/filter-data', methods=['POST'])
def filter_data():
    '''Handle data filtering requests'''
    try:
        # Get filter parameters
        filters = request.get_json()
        
        # Load and filter data
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'largeClean.csv'))
        
        # Apply filters
        if filters.get('date_from'):
            df = df[df['reportDate'] >= filters['date_from']]
        if filters.get('date_to'):
            df = df[df['reportDate'] <= filters['date_to']]
        if filters.get('load_type'):
            df = df[df['loadType'] == filters['load_type']]
        if filters.get('dropoff_site'):
            df = df[df['dropoffSite'] == filters['dropoff_site']]
        if filters.get('route_type'):
            df = df[df['routeType'] == filters['route_type']]
        
        # Limit results for performance
        df = df.head(1000)
        
        # Convert to JSON-serializable format
        result = df.to_dict('records')
        
        return jsonify({
            'data': result,
            'count': len(result),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-filtered', methods=['POST'])
def predict_filtered():
    '''Handle prediction requests for filtered data'''
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get filter parameters
        filters = request.get_json()
        
        # Load and filter data
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'largeClean.csv'))
        
        # Apply filters
        if filters.get('date_from'):
            df = df[df['reportDate'] >= filters['date_from']]
        if filters.get('date_to'):
            df = df[df['reportDate'] <= filters['date_to']]
        if filters.get('load_type'):
            df = df[df['loadType'] == filters['load_type']]
        if filters.get('dropoff_site'):
            df = df[df['dropoffSite'] == filters['dropoff_site']]
        if filters.get('route_type'):
            df = df[df['routeType'] == filters['route_type']]
        
        # Limit to a reasonable number for plotting (e.g., 100 points)
        # We'll take a sample if there are too many points
        if len(df) > 100:
            df = df.sample(n=100, random_state=42)
        
        # Preprocess the data
        X, y = preprocessor.transform(df)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Prepare result
        result = []
        for idx, row in df.iterrows():
            result.append({
                'actual': float(y.loc[idx]) if idx in y.index else 0.0,
                'predicted': float(predictions[idx]),
                'date': str(row['reportDate']),
                'loadType': str(row['loadType']),
                'dropoffSite': str(row['dropoffSite']),
                'routeType': str(row['routeType'])
            })
        
        return jsonify({
            'data': result,
            'count': len(result),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get-options')
def get_options():
    '''Get unique values for dropdown filters'''
    try:
        # Load sample data to get options
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'largeClean.csv'), nrows=50000)
        
        options = {
            'load_types': sorted(df['loadType'].dropna().unique().tolist()),
            'dropoff_sites': sorted(df['dropoffSite'].dropna().unique().tolist()),
            'route_types': sorted(df['routeType'].dropna().unique().tolist())
        }
        
        return jsonify(options)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    '''Get model information and feature importance'''
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': preprocessor.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Convert to list of dictionaries
        importance_list = feature_importance.to_dict('records')
        
        return jsonify({
            'feature_importance': importance_list,
            'model_type': 'Random Forest Regressor',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
