import time
from datetime import datetime, timezone
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
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_TAG = "xgb"

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

# 0 = high discharge on the same day as features; 1 = high discharge tomorrow
PREDICTION_HORIZON_DAYS = 1

ENABLE_HYPERPARAMETER_SEARCH = True
HYPERPARAMETER_SEARCH_ITERATIONS = 24
SEARCH_N_ESTIMATORS = 350
THRESHOLD_GRID_POINTS = 181

XGB_TUNABLE_KEYS = (
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "gamma",
    "subsample",
    "colsample_bytree",
)

XGB_MODEL_PARAMS = {
    "n_estimators": 900,
    "max_depth": 5,
    "learning_rate": 0.025,
    "min_child_weight": 4,
    "gamma": 0.2,
    "subsample": 0.88,
    "colsample_bytree": 0.88,
    "early_stopping_rounds": 50,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "scale_pos_weight_note": "computed per fold: (neg/pos) * SCALE_POS_WEIGHT_FACTOR",
}

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


def _future_discharge(df: pd.DataFrame, horizon: int) -> pd.Series:
    if horizon == 0:
        return df["river_discharge_m3s"]
    return df.groupby("location", group_keys=False)["river_discharge_m3s"].shift(
        -horizon
    )


def prepare_station_horizon_frames(
    station_train: pd.DataFrame,
    station_test: pd.DataFrame,
    quantile: float,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | tuple[None, None, None, None]:
    if horizon == 0:
        y_train = station_labels(station_train["river_discharge_m3s"], quantile)
        y_test = station_labels(station_test["river_discharge_m3s"], quantile)
        return station_train, station_test, y_train, y_test

    train_threshold = station_train["river_discharge_m3s"].quantile(quantile)
    tr = station_train.copy()
    te = station_test.copy()
    tr_target = _future_discharge(tr, horizon)
    te_target = _future_discharge(te, horizon)
    tr = tr.loc[tr_target.notna()]
    te = te.loc[te_target.notna()]
    tr_target = tr_target.loc[tr.index]
    te_target = te_target.loc[te.index]
    y_train = (tr_target > train_threshold).astype(int)
    y_test = (te_target > train_threshold).astype(int)
    return tr, te, y_train, y_test

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

def _scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    positives = int(np.asarray(y).sum())
    negatives = len(y) - positives
    return (negatives / max(positives, 1)) * SCALE_POS_WEIGHT_FACTOR


def _default_xgb_params() -> dict:
    return {key: XGB_MODEL_PARAMS[key] for key in XGB_TUNABLE_KEYS}


def _sample_xgb_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.choice([3, 4, 5, 6, 7])),
        "learning_rate": float(rng.choice([0.01, 0.025, 0.05])),
        "min_child_weight": int(rng.choice([2, 3, 4, 6, 8])),
        "gamma": float(rng.choice([0.0, 0.1, 0.2, 0.3])),
        "subsample": float(rng.choice([0.8, 0.85, 0.88, 0.9])),
        "colsample_bytree": float(rng.choice([0.8, 0.85, 0.88, 0.9])),
    }


def _make_xgb(
    xgb_params: dict,
    n_estimators: int,
    scale_pos_weight: float,
    early_stopping_rounds: int | None,
) -> XGBClassifier:
    kwargs: dict = {
        "n_estimators": n_estimators,
        "max_depth": xgb_params["max_depth"],
        "learning_rate": xgb_params["learning_rate"],
        "min_child_weight": xgb_params["min_child_weight"],
        "gamma": xgb_params["gamma"],
        "subsample": xgb_params["subsample"],
        "colsample_bytree": xgb_params["colsample_bytree"],
        "scale_pos_weight": scale_pos_weight,
        "random_state": XGB_MODEL_PARAMS["random_state"],
        "eval_metric": XGB_MODEL_PARAMS["eval_metric"],
    }
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**kwargs)


def hyperparameter_search(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    best_score = -1.0
    best_params = _sample_xgb_params(rng)
    y_val_np = y_val.to_numpy()
    if len(np.unique(y_val_np)) < 2:
        return best_params

    spw = _scale_pos_weight(y_fit)
    for _ in range(n_iter):
        params = _sample_xgb_params(rng)
        model = _make_xgb(
            params,
            SEARCH_N_ESTIMATORS,
            spw,
            early_stopping_rounds=XGB_MODEL_PARAMS["early_stopping_rounds"],
        )
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val_np, val_prob)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def train_station_model(
    X_fit,
    y_fit,
    X_val,
    y_val,
    xgb_params: dict | None = None,
) -> XGBClassifier:
    params = xgb_params if xgb_params is not None else _default_xgb_params()
    model = _make_xgb(
        params,
        XGB_MODEL_PARAMS["n_estimators"],
        _scale_pos_weight(y_fit),
        early_stopping_rounds=XGB_MODEL_PARAMS["early_stopping_rounds"],
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _flatten_xgb_params(params: dict) -> dict:
    return {f"xgb_{key}": value for key, value in params.items()}

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

    thresholds = np.linspace(0.05, 0.95, THRESHOLD_GRID_POINTS)
    for threshold in thresholds:
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
        for threshold in thresholds:
            pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_true, pred, zero_division=0)
            recall = recall_score(y_true, pred, zero_division=0)
            if precision >= min_precision and recall >= min_recall:
                score = fbeta_score(precision, recall, beta=0.5)
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)

    if best_score < 0:
        for threshold in thresholds:
            pred = (y_prob >= threshold).astype(int)
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

    low = max(0.01, best_threshold - 0.08)
    high = min(0.99, best_threshold + 0.08)
    for threshold in np.linspace(low, high, 81):
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
            )
        ):
            best_score = score
            best_precision = precision
            best_threshold = float(threshold)

    return best_threshold

#function to evaluate the model on the test set ofcc
def evaluate_station_split(
    station_train: pd.DataFrame,
    station_test: pd.DataFrame,
    quantile: float,
    min_precision: float,
    location: str,
    fold: int,
) -> dict | None:
    if len(station_train) < MIN_TRAIN_ROWS or len(station_test) == 0:
        return None

    tr, te, y_train, y_test = prepare_station_horizon_frames(
        station_train, station_test, quantile, PREDICTION_HORIZON_DAYS
    )
    if len(tr) < MIN_TRAIN_ROWS or len(te) == 0:
        return None

    if (
        y_train.sum() < MIN_TRAIN_POSITIVES
        or len(np.unique(y_test)) < 2
        or y_test.sum() < MIN_TEST_POSITIVES
    ):
        return None

    X_fit, y_fit, X_val, y_val = split_station_train_val(tr, y_train, VAL_FRACTION)
    if y_fit.sum() == 0 or len(np.unique(y_val)) < 2:
        return None

    t0 = time.perf_counter()
    if ENABLE_HYPERPARAMETER_SEARCH:
        xgb_params = hyperparameter_search(
            X_fit, y_fit, X_val, y_val, HYPERPARAMETER_SEARCH_ITERATIONS
        )
    else:
        xgb_params = _default_xgb_params()

    model = train_station_model(X_fit, y_fit, X_val, y_val, xgb_params=xgb_params)
    fit_prob = model.predict_proba(X_fit)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(te[FEATURES])[:, 1]

    if USE_PROBABILITY_CALIBRATION and len(np.unique(y_val)) >= 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_prob, y_val.to_numpy())
        fit_prob = calibrator.predict(fit_prob)
        val_prob = calibrator.predict(val_prob)
        test_prob = calibrator.predict(test_prob)

    threshold = pick_threshold(y_val.to_numpy(), val_prob, min_precision)
    effective_threshold = max(threshold, ALERT_PROB_FLOOR)
    train_f1 = f1_score(
        y_fit,
        (fit_prob >= effective_threshold).astype(int),
        zero_division=0,
    )
    test_pred = (test_prob >= effective_threshold).astype(int)

    if USE_PHYSICS_GATE:
        gate = physics_gate_mask(station_train, te)
        test_pred = (test_pred.astype(bool) & gate).astype(int)

    elapsed = time.perf_counter() - t0
    y_true = y_test.to_numpy()
    test_f1 = f1_score(y_true, test_pred, zero_division=0)
    print(f"  [{location} fold {fold}] tuned in {elapsed:.1f}s", flush=True)

    return {
        "threshold": effective_threshold,
        "best_xgb_params": xgb_params,
        "train_f1": train_f1,
        "train_test_f1_gap": train_f1 - test_f1,
        "roc_auc": roc_auc_score(y_true, test_prob),
        "accuracy": accuracy_score(y_true, test_pred),
        "precision": precision_score(y_true, test_pred, zero_division=0),
        "recall": recall_score(y_true, test_pred, zero_division=0),
        "f1": test_f1,
        "y_true": y_true,
        "y_pred": test_pred,
        "y_prob": test_prob,
    }

def _pipeline_settings() -> dict:
    return {
        "model_tag": MODEL_TAG,
        "dataset_slug": DATASET_SLUG,
        "csv_name": CSV_NAME,
        "risk_quantile": RISK_QUANTILE,
        "min_precision": MIN_PRECISION,
        "min_recall_when_tuning": MIN_RECALL_WHEN_TUNING,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "val_fraction": VAL_FRACTION,
        "scale_pos_weight_factor": SCALE_POS_WEIGHT_FACTOR,
        "use_probability_calibration": USE_PROBABILITY_CALIBRATION,
        "use_physics_gate": USE_PHYSICS_GATE,
        "physics_precip7_train_quantile": PHYSICS_PRECIP7_TRAIN_QUANTILE,
        "physics_discharge_lag1_train_quantile": PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE,
        "alert_prob_floor": ALERT_PROB_FLOOR,
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "enable_hyperparameter_search": ENABLE_HYPERPARAMETER_SEARCH,
        "hyperparameter_search_iterations": HYPERPARAMETER_SEARCH_ITERATIONS,
        "search_n_estimators": SEARCH_N_ESTIMATORS,
        "threshold_grid_points": THRESHOLD_GRID_POINTS,
        "xgb_model_params": XGB_MODEL_PARAMS,
    }


def _write_key_values(fh, mapping: dict, prefix: str = "") -> None:
    for key, value in mapping.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _write_key_values(fh, value, prefix=f"{label}.")
        else:
            fh.write(f"{label} = {value}\n")


def save_run_artifacts(
    fold_rows: list[dict],
    macro: dict,
    micro: dict,
    false_alarms: int,
    alerts: int,
    pooled_true: list[int],
    pooled_pred: list[int],
    pooled_prob: list[float],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_rows = [
        {"level": "macro", **macro},
        {"level": "micro", **micro},
        {
            "level": "operational",
            "false_alarms": false_alarms,
            "alerts": alerts,
            "false_alarm_rate": false_alarms / alerts if alerts else 0.0,
        },
    ]
    metrics_df = pd.concat(
        [pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)],
        ignore_index=True,
    )
    metrics_path = RESULTS_DIR / f"{MODEL_TAG}_metrics_{run_ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    pred_df = pd.DataFrame(
        {"y_true": pooled_true, "y_pred": pooled_pred, "y_prob": pooled_prob}
    )
    pred_path = RESULTS_DIR / f"{MODEL_TAG}_predictions_{run_ts}.csv"
    pred_df.to_csv(pred_path, index=False)

    settings = _pipeline_settings()
    settings["run_timestamp_utc"] = run_ts
    settings["summary_macro"] = macro
    settings["summary_micro"] = micro
    settings["false_alarms"] = false_alarms
    settings["alerts"] = alerts
    best_f1_row = max(fold_rows, key=lambda r: r.get("f1", -1)) if fold_rows else None
    settings_path = RESULTS_DIR / f"{MODEL_TAG}_settings_{run_ts}.txt"
    with settings_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Model run: {MODEL_TAG}\n")
        fh.write(f"Timestamp (UTC): {run_ts}\n\n")
        fh.write("=== Pipeline / threshold settings ===\n")
        for key, value in settings.items():
            if key in ("xgb_model_params", "summary_macro", "summary_micro"):
                continue
            fh.write(f"{key} = {value}\n")
        fh.write("\n=== XGB training defaults (n_estimators / early stopping) ===\n")
        _write_key_values(fh, XGB_MODEL_PARAMS, prefix="xgb")
        fh.write(
            "\n(Per-fold tuned max_depth, lr, etc. are columns xgb_* in the metrics CSV.)\n"
        )
        fh.write("\n=== Best fold by test F1 ===\n")
        if best_f1_row:
            _write_key_values(
                fh,
                {k: v for k, v in best_f1_row.items() if k != "level"},
                prefix="best_fold",
            )
        else:
            fh.write("best_fold = (none)\n")
        fh.write("\n=== Summary metrics (macro = mean over station-folds) ===\n")
        _write_key_values(fh, macro, prefix="macro")
        _write_key_values(fh, micro, prefix="micro")
        fh.write(f"operational_false_alarms = {false_alarms}\n")
        fh.write(f"operational_alerts = {alerts}\n")
        fh.write(
            f"operational_false_alarm_rate = "
            f"{false_alarms / alerts if alerts else 0.0:.6f}\n"
        )
        fh.write(
            f"\nMetrics CSV: {metrics_path.name}\n"
            f"Predictions CSV: {pred_path.name}\n"
        )

    latest_settings = RESULTS_DIR / f"{MODEL_TAG}_settings_latest.txt"
    latest_metrics = RESULTS_DIR / f"{MODEL_TAG}_metrics_latest.csv"
    latest_settings.write_text(settings_path.read_text(encoding="utf-8"), encoding="utf-8")
    metrics_df.to_csv(latest_metrics, index=False)

    print(f"\nSaved results to {RESULTS_DIR}/")
    print(f"  {metrics_path.name}")
    print(f"  {pred_path.name}")
    print(f"  {settings_path.name}")


#main function to run the model
def main() -> None:
    df = add_location_features(load_data())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    #lists to store the metrics for the model
    split_metrics: list[dict] = []
    fold_rows: list[dict] = []
    scorable_metrics: list[dict] = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_prob: list[float] = []

    print(
        f"Rows: {len(df)} | Stations: {df['location'].nunique()} | "
        f"Per-station time-series CV | Horizon: {PREDICTION_HORIZON_DAYS} day(s) ahead | "
        f"Risk quantile: {RISK_QUANTILE} | Min precision: {MIN_PRECISION} | "
        f"XGB search: {ENABLE_HYPERPARAMETER_SEARCH} "
        f"({HYPERPARAMETER_SEARCH_ITERATIONS} iters/fold) | "
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
                location,
                fold + 1,
            )
            if result is None:
                continue

            split_metrics.append(result)
            fold_row = {
                "level": "fold",
                "location": location,
                "fold": fold + 1,
                "threshold": result["threshold"],
                "train_f1": result["train_f1"],
                "train_test_f1_gap": result["train_test_f1_gap"],
                "roc_auc": result["roc_auc"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
            fold_row.update(_flatten_xgb_params(result["best_xgb_params"]))
            fold_rows.append(fold_row)
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
                f"F1={result['f1'] * 100:.1f}% | "
                f"train→test F1 gap={result['train_test_f1_gap'] * 100:+.1f}pp"
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
    save_run_artifacts(
        fold_rows,
        macro,
        micro,
        false_alarms,
        alerts,
        pooled_true,
        pooled_pred,
        pooled_prob,
    )


if __name__ == "__main__":
    main()
