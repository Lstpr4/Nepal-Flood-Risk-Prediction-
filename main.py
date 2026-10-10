import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026")
csv =  pd.read_csv(path)

flood_threshold  =  csv['riverdischarge_m3s'].quantile(0.95)
csv['is_high_flood_risk'] = (csv['riverdischarge_m3s'] > flood_threshold).astype(int)
X = csv[[
    'precipitation_mm',
    'soil_moisture_0_100cm_m3m3',
    'rain_mm',
    'temperature_mean_c',
    'relative_humidity_mean_pct',
    'wind_speed_max_kmh',
    ]]
y = (csv['riverdischarge_m3s'] > flood_threshold).astype(int)
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# 6. Initialize and Train the XGBoost Classifier
# scale_pos_weight helps handle class imbalance since floods are rare events (~10%)
neg_pos_ratio = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(
    n_estimators=[100, 200, 300, 400],
    max_depth=[3, 4, 5, 6, 7, 8, 9, 10],
    learning_rate=[0.01, 0.05, 0.1, 0.2],
    scale_pos_weight=neg_pos_ratio,
    early_stopping_rounds=10,
    random_state=42,
)

model.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f'--- Evaluation on Final Time-Series Split Fold ---')
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
