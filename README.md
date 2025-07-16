# UFC Fight Predictor - ML Pipeline Website

A beautiful, interactive web application showcasing an advanced machine learning pipeline for predicting UFC fight outcomes. The website displays historical fight statistics, visualizations, and real-time predictions for upcoming fights.

## 🥊 Features

- **Interactive Visualizations**: Charts showing fight statistics, weight class distribution, and finish types
- **Real-time Predictions**: Live predictions for upcoming UFC fights using a trained ML model
- **Modern UI**: Responsive design with beautiful gradients and animations
- **Model Information**: Detailed breakdown of the ML pipeline architecture
- **API Endpoints**: RESTful API for accessing predictions and statistics

## 🏗️ Architecture

### ML Model
- **Algorithm**: Multi-Layer Perceptron (MLP) Neural Network
- **Features**: 27 engineered features including:
  - Betting odds and expected values
  - Fighter statistics (wins, losses, streaks)
  - Physical attributes (height, reach, age differences)
  - Performance metrics (significant strikes, takedowns)
- **Performance**: 84.6% accuracy on recent predictions
- **Training Data**: 6,528 historical UFC fights

### Web Application
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualizations**: Plotly.js
- **Icons**: Font Awesome

## 📊 Data Sources

- **Historical Data**: `data/ufc-master.csv` - Complete UFC fight history
- **Upcoming Fights**: `data/upcoming.csv` - Scheduled fights for prediction
- **Trained Model**: `models/mlp_model.pkl` - Serialized ML model

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   cd ufc-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
ufc-project/
├── app.py                 # Flask web application
├── pipeline.py            # Data preprocessing pipeline
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data files
│   ├── ufc-master.csv    # Historical fight data
│   └── upcoming.csv      # Upcoming fights
├── models/               # Trained models
│   └── mlp_model.pkl     # MLP neural network
├── templates/            # HTML templates
│   └── index.html        # Main webpage
└── notebooks/            # Jupyter notebooks
    ├── analysis.py       # Data analysis
    ├── visualizations.ipynb
    ├── modeling.ipynb
    └── predictions.ipynb
```

## 🔧 API Endpoints

### GET `/api/stats`
Returns historical fight statistics including:
- Total number of fights
- Unique fighters
- Winner distribution (Red vs Blue corner)
- Weight class distribution
- Fight finish types

**Response:**
```json
{
  "success": true,
  "stats": {
    "totalFights": 6528,
    "uniqueFighters": 1234,
    "redWins": 3200,
    "blueWins": 3328,
    "weightClasses": {
      "labels": ["Lightweight", "Welterweight", ...],
      "values": [850, 720, ...]
    },
    "finishTypes": {
      "labels": ["U-DEC", "KO/TKO", ...],
      "values": [1200, 980, ...]
    }
  }
}
```

### GET `/api/predictions`
Returns predictions for upcoming fights including:
- Fighter names
- Predicted winner (Red/Blue corner)
- Confidence percentage
- Weight class

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "red": "Colby Covington",
      "blue": "Joaquin Buckley",
      "prediction": "Blue",
      "confidence": 0.67,
      "weightClass": "Welterweight"
    }
  ]
}
```

## 📈 Model Performance

The MLP model achieves:
- **Accuracy**: 84.6% on recent predictions
- **Training Data**: 6,528 fights with 27 features
- **Cross-validation**: 5-fold CV for hyperparameter tuning
- **Feature Engineering**: Advanced feature creation from raw fight data

## 🎨 Visualizations

The website includes interactive charts:
1. **Winner Distribution**: Red vs Blue corner wins
2. **Weight Class Distribution**: Fights across different weight classes
3. **Fight Finish Types**: Distribution of fight outcomes (KO, submission, decision, etc.)

## 🔮 Predictions

The prediction system:
- Processes upcoming fight data through the same pipeline used for training
- Applies the trained MLP model to generate win probabilities
- Displays predictions with confidence scores
- Color-codes predictions (Red/Blue) for easy visualization

## 🛠️ Development

### Adding New Fights
1. Add fight data to `data/upcoming.csv`
2. Ensure all required features are present
3. Restart the Flask application
4. Predictions will automatically update

### Modifying the Model
1. Update the model in `notebooks/modeling.ipynb`
2. Save the new model to `models/`
3. Update `app.py` to load the new model

### Customizing the UI
- Modify `templates/index.html` for layout changes
- Update CSS styles in the `<style>` section
- Add new charts by extending the JavaScript functions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- UFC for providing fight data
- Scikit-learn for ML algorithms
- Plotly for interactive visualizations
- Font Awesome for icons

---

**Note**: This is a demonstration project. Fight predictions should not be used for gambling purposes. 