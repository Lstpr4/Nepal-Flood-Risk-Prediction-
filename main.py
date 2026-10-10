import kagglehub
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

path = kagglehub.dataset_download("tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026")
csv =  pd.read_csv(Path(path) / "CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv")

# Ensure chronological order by date
csv['date'] = pd.to_datetime(csv['date'])
csv = csv.sort_values('date').reset_index(drop=True)


flood_threshold  =  csv['river_discharge_m3s'].quantile(0.95)
csv['is_high_flood_risk'] = (csv['river_discharge_m3s'] > flood_threshold).astype(int)


features = [
    'precipitation_mm',
    'soil_moisture_0_100cm_m3m3',
    'rain_mm',
    'temperature_mean_c',
    'relative_humidity_mean_pct',
    'wind_speed_max_kmh',
]

X = csv[features]
y = csv['is_high_flood_risk']
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, test_index) in enumerate(tscv.split(X)): 
  if fold > 0:
    break
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# 6. Initialize and Train the XGBoost Classifier
# scale_pos_weight helps handle class imbalance since floods are rare events (~10%)
neg_pos_ratio = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(
  n_estimators=200,
  max_depth=10,
  learning_rate=0.1,
  scale_pos_weight=neg_pos_ratio,
  early_stopping_rounds=10,
  random_state=42,
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Allows us to predict and find the probabilities of the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob)* 100 :.2f}%')
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100 :.2f}%')
print(f'Precision: {precision_score(y_test, y_pred)*100:.2f}%')
print(f'Recall: {recall_score(y_test, y_pred)*100:.2f}%')
print(f'F1 Score: {f1_score(y_test, y_pred)*100:2f}%')
