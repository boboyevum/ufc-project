import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('data/df_modeling.csv')
print(f"Dataset shape: {df.shape}")

X = df.drop('Winner', axis=1)
y = df['Winner']

print(f"Features: {X.shape[1]}")
print(f"Target distribution: {y.value_counts()}")

train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
y_val = y.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]
y_test = y.iloc[train_size + val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n=== FEATURE SELECTION ===")
selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

print("\n=== ADVANCED MODELS ===")

rf_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

gb_params = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [6, 8, 10],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5]
}

svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly'],
    'class_weight': ['balanced']
}

mlp_params = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500],
    'early_stopping': [True]
}

models = {
    'RandomForest': (RandomForestClassifier(random_state=42), rf_params),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), gb_params),
    'SVM': (SVC(random_state=42), svm_params),
    'MLP': (MLPClassifier(random_state=42), mlp_params)
}

best_models = {}
val_scores = {}

for name, (model, params) in models.items():
    print(f"\nTuning {name}...")
    clf = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    clf.fit(X_train_selected, y_train)
    
    best_models[name] = clf.best_estimator_
    val_pred = clf.predict(X_val_selected)
    val_score = accuracy_score(y_val, val_pred)
    val_scores[name] = val_score
    
    print(f"{name} - Best params: {clf.best_params_}")
    print(f"{name} - Validation accuracy: {val_score:.4f}")

print("\n=== ENSEMBLE MODEL ===")
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)

ensemble.fit(X_train_selected, y_train)
ensemble_val_pred = ensemble.predict(X_val_selected)
ensemble_val_score = accuracy_score(y_val, ensemble_val_pred)
print(f"Ensemble validation accuracy: {ensemble_val_score:.4f}")

print("\n=== FINAL EVALUATION ===")
test_pred = ensemble.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

print("\n=== INDIVIDUAL MODEL PERFORMANCE ===")
for name, model in best_models.items():
    test_pred = model.predict(X_test_selected)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"{name}: {test_acc:.4f}")

print("\n=== FEATURE IMPORTANCE ===")
rf_model = best_models['RandomForest']
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
print(feature_importance.head(10))

print("\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_test, test_pred))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, test_pred)
print(cm) 