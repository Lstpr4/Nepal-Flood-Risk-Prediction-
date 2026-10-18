"""Shared data, labels, and evaluation helpers for all model pipelines."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATASET_SLUG = "tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026"
CSV_NAME = "CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv"

RISK_QUANTILE = 0.85
MIN_PRECISION = 0.65
MIN_RECALL_WHEN_TUNING = 0.58
N_SPLITS = 3
RANDOM_STATE = 42
VAL_FRACTION = 0.25
MIN_TRAIN_ROWS = 150
MIN_TRAIN_POSITIVES = 5
MIN_TEST_POSITIVES = 3

USE_PROBABILITY_CALIBRATION = True
USE_PHYSICS_GATE = True
PHYSICS_PRECIP7_TRAIN_QUANTILE = 0.60
PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE = 0.65
ALERT_PROB_FLOOR = 0.68
PREDICTION_HORIZON_DAYS = 1
THRESHOLD_GRID_POINTS = 181

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


def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download(DATASET_SLUG)
    df = pd.read_csv(Path(path) / CSV_NAME)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["location", "date"]).reset_index(drop=True)


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("location", group_keys=False)

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
    out[FEATURES] = out[FEATURES].fillna(0)
    return out


def station_labels(series: pd.Series, quantile: float) -> pd.Series:
    threshold = series.quantile(quantile)
    return (series > threshold).astype(int)


def _future_discharge(df: pd.DataFrame, horizon: int) -> pd.Series:
    if horizon == 0:
        return df["river_discharge_m3s"]
    return df.groupby("location", group_keys=False)["river_discharge_m3s"].shift(-horizon)


def prepare_station_horizon_frames(
    station_train: pd.DataFrame,
    station_test: pd.DataFrame,
    quantile: float,
    horizon: int,
):
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


def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def physics_gate_mask(station_train: pd.DataFrame, station_test: pd.DataFrame) -> np.ndarray:
    precip_floor = station_train["precip_sum7"].quantile(PHYSICS_PRECIP7_TRAIN_QUANTILE)
    discharge_floor = station_train["discharge_lag1"].quantile(
        PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE
    )
    heavy_rain = station_test["precip_sum7"].to_numpy() >= precip_floor
    elevated_flow = station_test["discharge_lag1"].to_numpy() >= discharge_floor
    return heavy_rain | elevated_flow


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
    min_recall: float = MIN_RECALL_WHEN_TUNING,
) -> float:
    thresholds = np.linspace(0.05, 0.95, THRESHOLD_GRID_POINTS)
    best_threshold = 0.5
    best_score = -1.0
    best_precision = -1.0

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
