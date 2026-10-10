from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

#the kaggle dataset I am using
DATASET_SLUG = "tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026"
CSV_NAME = "CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv"

#the quantile of the river discharge that is considered a flood, 
RISK_QUANTILE = 0.85

#the minimum precision for the model to be considered a flood
MIN_PRECISION = 0.65
#the minimum recall for the model to be considered a flood
MIN_RECALL_WHEN_TUNING = 0.58
#the number of splits for the time series cross validation
N_SPLITS = 3
#so I can reproduce the results
RANDOM_STATE = 42

#the fraction of the data that is used for validation
VAL_FRACTION = 0.25
#the minimum number of rows for the model to be considered a flood
MIN_TRAIN_ROWS = 150
#the minimum number of positive rows for the model to be considered a flood
MIN_TRAIN_POSITIVES = 5
#the minimum number of positive rows for the model to be considered a flood
MIN_TEST_POSITIVES = 3

# Slightly below 1.0 → model is less eager to flag floods (fewer false positives)
SCALE_POS_WEIGHT_FACTOR = 1.0
USE_PROBABILITY_CALIBRATION = True
# Require recent rain OR elevated yesterday discharge before issuing an alert
USE_PHYSICS_GATE = True
PHYSICS_PRECIP7_TRAIN_QUANTILE = 0.60
PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE = 0.65
#never alert below this probability (cuts false alarms; lowers recall slightly)
ALERT_PROB_FLOOR = 0.68

#features used to train the model
FEATURES = [
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "rain_mm",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh",
    "precipitation_hours",
    "wind_gusts_max_kmh",
    "dew_point_mean_c",
    "elevation_m",
    "precip_lag1",
    "precip_lag2",
    "precip_lag3",
    "precip_lag7",
    "precip_lag14",
    "precip_sum3",
    "precip_sum7",
    "precip_sum14",
    "discharge_lag1",
    "discharge_lag2",
    "discharge_lag3",
    "discharge_lag7",
    "discharge_lag14",
    "discharge_roll7_mean",
    "discharge_roll14_mean",
    "discharge_to_roll7",
    "saturation_rain_interaction",
]

#function to load my dataset from kaggle and sort it by location and date
def load_data() -> pd.DataFrame:
  path = kagglehub.dataset_download(DATASET_SLUG)
  df = pd.read_csv(Path(path) / CSV_NAME)
  df["date"] = pd.to_datetime(df["date"])
  return df.sort_values(["location", "date"]).reset_index(drop=True)


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
  #copies dataframe to avoid messing up my OG dataframe
  out = df.copy()
  grouped = out.groupby("location", group_keys=False)

  #creates features for the precipitation and discharge lags
  for lag in (1, 2, 3, 7, 14):
    out[f"precip_lag{lag}"] = grouped["precipitation_mm"].shift(lag)
    out[f"discharge_lag{lag}"] = grouped["river_discharge_m3s"].shift(lag)

  for window in (3, 7, 14):
    out[f"precip_sum{window}"] = grouped["precipitation_mm"].transform(
        lambda s: s.rolling(window, min_periods=1).sum()
    )

  out["discharge_roll7_mean"] = grouped["river_discharge_m3s"].transform(
      lambda s: s.rolling(7, min_periods=1).mean()
  )
  out["discharge_roll14_mean"] = grouped["river_discharge_m3s"].transform(
      lambda s: s.rolling(14, min_periods=1).mean()
  )
  out["discharge_to_roll7"] = out["discharge_lag1"] / (out["discharge_roll7_mean"] + 1e-3)
  out["saturation_rain_interaction"] = (
  out["precipitation_mm"] * out["soil_moisture_0_100cm_m3m3"]
  )
  #fills the missing values with 0
  out[FEATURES] = out[FEATURES].fillna(0)
  return out

#function just creates labels for the model to train
def station_labels(series: pd.Series, quantile: float) -> pd.Series:
  threshold = series.quantile(quantile)
  return (series > threshold).astype(int)

#function to split the data into training and validation sets
def split_station_train_val(
  station_train: pd.DataFrame,
  y_train: pd.Series,
  val_fraction: float,
):
    unique_dates = np.sort(station_train["date"].unique())
    n_val = max(20, int(len(unique_dates) * val_fraction))
    val_dates = set(unique_dates[-n_val:])
    val_mask = station_train["date"].isin(val_dates)

    X = station_train[FEATURES]
    return (
        X.loc[~val_mask],
        y_train.loc[~val_mask],
        X.loc[val_mask],
        y_train.loc[val_mask],
    )

#The name speaks for itself..
def train_station_model(X_fit, y_fit, X_val, y_val) -> XGBClassifier:
    positives = int(y_fit.sum())
    negatives = len(y_fit) - positives
    scale_pos_weight = (negatives / max(positives, 1)) * SCALE_POS_WEIGHT_FACTOR

    #the model we are using to train the model
    model = XGBClassifier(
        n_estimators=900,#number of trees in the model
        max_depth=5,
        learning_rate=0.025,
        min_child_weight=4,#minimum weight of a leaf node
        gamma=0.2,#minimum loss reduction required to make a split
        subsample=0.88,#fraction of the data to be used for training the model
        colsample_bytree=0.88,#fraction of the features to be used for training the model
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        eval_metric="auc",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model

#used to calculate the fbeta score
def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta #beta is used to balance the precision and recall
    return (1 + b2) * precision * recall / (b2 * precision + recall)

#used to calibrate the probabilities of the model
def calibrate_probabilities(
    y_val: np.ndarray, val_prob: np.ndarray, test_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(y_val)) < 2:
        return val_prob, test_prob
    #isotonic regression is used to calibrate the probabilities of the model
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_prob, y_val)
    return calibrator.predict(val_prob), calibrator.predict(test_prob)


def physics_gate_mask(station_train: pd.DataFrame, station_test: pd.DataFrame) -> np.ndarray:
    precip_floor = station_train["precip_sum7"].quantile(PHYSICS_PRECIP7_TRAIN_QUANTILE)
    discharge_floor = station_train["discharge_lag1"].quantile(
        PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE
    )
    heavy_rain = station_test["precip_sum7"].to_numpy() >= precip_floor
    elevated_flow = station_test["discharge_lag1"].to_numpy() >= discharge_floor
    return heavy_rain | elevated_flow

#would pick the threshold for the model to predict the floods
def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
    min_recall: float = MIN_RECALL_WHEN_TUNING,
) -> float:
    best_threshold = 0.5
    best_score = -1.0
    best_precision = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        score = fbeta_score(precision, recall, beta=0.5)

        if (
            precision >= min_precision
            and recall >= min_recall
            and (
                score > best_score
                or (score == best_score and precision > best_precision)
                or (
                    score == best_score
                    and precision == best_precision
                    and threshold > best_threshold
                )
            )
        ):
            best_score = score
            best_precision = precision
            best_threshold = float(threshold)

    if best_score < 0:
        #linear space is used to create a list of thresholds
        for threshold in np.linspace(0.05, 0.95, 91):
            pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, pred, zero_division=0)
            recall = recall_score(y_true, pred, zero_division=0)
            if precision >= min_precision and recall >= min_recall:
                score = fbeta_score(precision, recall, beta=0.5)
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)

    if best_score < 0:
        for threshold in np.linspace(0.05, 0.95, 91):
            pred = (y_prob >= threshold).astype(int)
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

    return best_threshold

#function to evaluate the model on the test set ofcc
def evaluate_station_split(
    station_train: pd.DataFrame,
    station_test: pd.DataFrame,
    quantile: float,
    min_precision: float,
) -> dict | None:
    if len(station_train) < MIN_TRAIN_ROWS or len(station_test) == 0:
        return None
    #creates labels for the training and test sets
    y_train = station_labels(station_train["river_discharge_m3s"], quantile)
    y_test = station_labels(station_test["river_discharge_m3s"], quantile)

    if (
        y_train.sum() < MIN_TRAIN_POSITIVES
        or len(np.unique(y_test)) < 2
        or y_test.sum() < MIN_TEST_POSITIVES
    ):
        return None

    X_fit, y_fit, X_val, y_val = split_station_train_val(
        station_train, y_train, VAL_FRACTION
    )
    if y_fit.sum() == 0 or len(np.unique(y_val)) < 2:
        return None

    model = train_station_model(X_fit, y_fit, X_val, y_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(station_test[FEATURES])[:, 1]

    if USE_PROBABILITY_CALIBRATION:
        val_prob, test_prob = calibrate_probabilities(
            y_val.to_numpy(), val_prob, test_prob
        )

    threshold = pick_threshold(y_val.to_numpy(), val_prob, min_precision)
    effective_threshold = max(threshold, ALERT_PROB_FLOOR)
    test_pred = (test_prob >= effective_threshold).astype(int)

    if USE_PHYSICS_GATE:
        gate = physics_gate_mask(station_train, station_test)
        test_pred = (test_pred.astype(bool) & gate).astype(int)

    y_true = y_test.to_numpy()
    return {
        "threshold": effective_threshold,
        "roc_auc": roc_auc_score(y_true, test_prob),
        "accuracy": accuracy_score(y_true, test_pred),
        "precision": precision_score(y_true, test_pred, zero_division=0),
        "recall": recall_score(y_true, test_pred, zero_division=0),
        "f1": f1_score(y_true, test_pred, zero_division=0),
        "y_true": y_true,
        "y_pred": test_pred,
        "y_prob": test_prob,
    }

#main function to run the model
def main() -> None:
    df = add_location_features(load_data())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    #lists to store the metrics for the model
    split_metrics: list[dict] = []
    scorable_metrics: list[dict] = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_prob: list[float] = []

    print(
        f"Rows: {len(df)} | Stations: {df['location'].nunique()} | "
        f"Per-station time-series CV | Risk quantile: {RISK_QUANTILE} | "
        f"Min precision: {MIN_PRECISION} | "
        f"Calibration: {USE_PROBABILITY_CALIBRATION} | "
        f"Physics gate: {USE_PHYSICS_GATE} | "
        f"Alert prob floor: {ALERT_PROB_FLOOR}\n"
    )

    for location in sorted(df["location"].unique()):
        station_df = df[df["location"] == location].sort_values("date")
        station_dates = station_df["date"].unique()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(station_dates)):
            train_dates = station_dates[train_idx]
            test_dates = station_dates[test_idx]

            station_train = station_df[station_df["date"].isin(train_dates)]
            station_test = station_df[station_df["date"].isin(test_dates)]

            result = evaluate_station_split(
                station_train,
                station_test,
                RISK_QUANTILE,
                MIN_PRECISION,
            )
            if result is None:
                continue

            split_metrics.append(result)
            failed = result["f1"] == 0 and result["y_true"].sum() >= MIN_TEST_POSITIVES
            if not failed:
                scorable_metrics.append(result)
            pooled_true.extend(result["y_true"].tolist())
            pooled_pred.extend(result["y_pred"].tolist())
            pooled_prob.extend(result["y_prob"].tolist())

            print(
                f"{location} fold {fold + 1}: thr={result['threshold']:.2f} | "
                f"Prec={result['precision'] * 100:.1f}% | "
                f"Rec={result['recall'] * 100:.1f}% | "
                f"F1={result['f1'] * 100:.1f}%"
            )

    if not split_metrics:
        print("No valid station folds to report.")
        return

    macro_source = scorable_metrics if scorable_metrics else split_metrics
    macro = {
        key: float(np.mean([m[key] for m in macro_source]))
        for key in ("roc_auc", "accuracy", "precision", "recall", "f1")
    }

    y_true = np.array(pooled_true)
    y_pred = np.array(pooled_pred)
    y_prob = np.array(pooled_prob)
    micro = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    false_alarms = int(((y_pred == 1) & (y_true == 0)).sum())
    alerts = int((y_pred == 1).sum())
    false_alarm_rate = false_alarms / alerts if alerts else 0.0

    print(
        f"\nEvaluated {len(split_metrics)} station-folds "
        f"({len(macro_source)} used for macro mean, "
        f"{len(pooled_true)} test rows pooled)\n"
    )
    print("--- Macro mean (average of each station-fold) ---")
    print(
        f"ROC-AUC={macro['roc_auc'] * 100:.2f}% | "
        f"Acc={macro['accuracy'] * 100:.2f}% | "
        f"Prec={macro['precision'] * 100:.2f}% | "
        f"Rec={macro['recall'] * 100:.2f}% | "
        f"F1={macro['f1'] * 100:.2f}%"
    )
    print("--- Micro (all pooled test predictions) ---")
    print(
        f"ROC-AUC={micro['roc_auc'] * 100:.2f}% | "
        f"Acc={micro['accuracy'] * 100:.2f}% | "
        f"Prec={micro['precision'] * 100:.2f}% | "
        f"Rec={micro['recall'] * 100:.2f}% | "
        f"F1={micro['f1'] * 100:.2f}%"
    )
    print(
        f"False alarms: {false_alarms} / {alerts} alerts "
        f"({false_alarm_rate * 100:.1f}% of alerts were wrong)"
    )
if __name__ == "__main__":
    main()
