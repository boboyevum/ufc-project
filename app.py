from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import sys
import os
from pipeline import data_pipeline

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/mlp_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    try:
        # Load and process upcoming fights data
        df_upcoming, red_corner, blue_corner = data_pipeline('upcoming.csv')
        
        # Make predictions
        predictions = model.predict(df_upcoming)
        probabilities = model.predict_proba(df_upcoming)[:, 1]  # Red corner win probability
        
        # Create predictions list
        fight_predictions = []
        for i, (red, blue, pred, prob) in enumerate(zip(red_corner, blue_corner, predictions, probabilities)):
            # Get weight class from original data
            upcoming_data = pd.read_csv('data/upcoming.csv')
            weight_class = upcoming_data.iloc[i]['WeightClass'] if i < len(upcoming_data) else 'Unknown'
            
            fight_predictions.append({
                'red': red,
                'blue': blue,
                'prediction': 'Red' if pred == 1 else 'Blue',
                'confidence': float(prob if pred == 1 else 1 - prob),
                'weightClass': weight_class
            })
        
        return jsonify({
            'success': True,
            'predictions': fight_predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    try:
        # Load main dataset for statistics
        ufc_data = pd.read_csv('data/ufc-master.csv')
        
        # Calculate statistics
        total_fights = len(ufc_data)
        unique_fighters = len(set(ufc_data['RedFighter'].tolist() + ufc_data['BlueFighter'].tolist()))
        
        # Winner distribution
        winner_counts = ufc_data['Winner'].value_counts()
        red_wins = winner_counts.get('Red', 0)
        blue_wins = winner_counts.get('Blue', 0)
        
        # Weight class distribution
        weight_class_counts = ufc_data['WeightClass'].value_counts().head(8)
        
        # Finish types
        finish_counts = ufc_data['Finish'].value_counts().head(8)
        
        return jsonify({
            'success': True,
            'stats': {
                'totalFights': total_fights,
                'uniqueFighters': unique_fighters,
                'redWins': int(red_wins) if red_wins is not None else 0,
                'blueWins': int(blue_wins) if blue_wins is not None else 0,
                'weightClasses': {
                    'labels': weight_class_counts.index.tolist(),
                    'values': weight_class_counts.values.tolist()
                },
                'finishTypes': {
                    'labels': finish_counts.index.tolist(),
                    'values': finish_counts.values.tolist()
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 