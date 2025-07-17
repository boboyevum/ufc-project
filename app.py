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
                'weightClass': weight_class,
                'actualResult': 'TBD'  # Since these are upcoming fights
            })
        
        return jsonify({
            'success': True,
            'predictions': fight_predictions
        })
    
    except Exception as e:
        print(f"Error in predictions API: {str(e)}")
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
        
        # Gender distribution
        gender_counts = ufc_data['Gender'].value_counts()
        
        # Title bout analysis
        title_bout_counts = ufc_data['TitleBout'].value_counts()
        
        # Stance analysis
        red_stance_counts = ufc_data['RedStance'].value_counts().head(3)
        blue_stance_counts = ufc_data['BlueStance'].value_counts().head(3)
        
        # Age analysis (if available)
        age_data = {}
        if 'RedAge' in ufc_data.columns and 'BlueAge' in ufc_data.columns:
            red_ages = ufc_data['RedAge'].dropna()
            blue_ages = ufc_data['BlueAge'].dropna()
            age_data = {
                'redAvgAge': float(red_ages.mean()),
                'blueAvgAge': float(blue_ages.mean()),
                'redAgeRange': [float(red_ages.min()), float(red_ages.max())],
                'blueAgeRange': [float(blue_ages.min()), float(blue_ages.max())]
            }
        
        # Win streak analysis
        win_streak_data = {}
        if 'RedCurrentWinStreak' in ufc_data.columns and 'BlueCurrentWinStreak' in ufc_data.columns:
            red_streaks = ufc_data['RedCurrentWinStreak'].dropna()
            blue_streaks = ufc_data['BlueCurrentWinStreak'].dropna()
            if len(red_streaks) > 0 and len(blue_streaks) > 0:
                red_max = red_streaks.max()
                blue_max = blue_streaks.max()
                win_streak_data = {
                    'redAvgStreak': float(red_streaks.mean()),
                    'blueAvgStreak': float(blue_streaks.mean()),
                    'redMaxStreak': int(red_max) if red_max is not None else 0,
                    'blueMaxStreak': int(blue_max) if blue_max is not None else 0
                }
        
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
                },
                'genderDistribution': {
                    'labels': gender_counts.index.tolist(),
                    'values': gender_counts.values.tolist()
                },
                'titleBouts': {
                    'labels': ['Title Fights', 'Non-Title Fights'],
                    'values': [int(title_bout_counts.get(True, 0)), int(title_bout_counts.get(False, 0))]
                },
                'redStance': {
                    'labels': red_stance_counts.index.tolist(),
                    'values': red_stance_counts.values.tolist()
                },
                'blueStance': {
                    'labels': blue_stance_counts.index.tolist(),
                    'values': blue_stance_counts.values.tolist()
                },
                'ageData': age_data,
                'winStreakData': win_streak_data
            }
        })
    
    except Exception as e:
        print(f"Error in stats API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 