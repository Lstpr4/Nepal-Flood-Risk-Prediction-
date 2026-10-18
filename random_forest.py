import time
from datetime import datetime, timezone #used to get the current time
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier #Model we will be using
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from pipeline_common import PREDICTION_HORIZON_DAYS, prepare_station_horizon_frames

#base directory of the project, used to save the results
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_TAG = "random_forest"

#kaggle dataset we will be using
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

#hyperparameter search iterations
HYPERPARAMETER_SEARCH_ITERATIONS = 28
SEARCH_N_ESTIMATORS = 280
FINAL_N_ESTIMATORS = 650
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

#function to load the data from the kaggle dataset
def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download(DATASET_SLUG)
    df = pd.read_csv(Path(path) / CSV_NAME)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["location", "date"]).reset_index(drop=True)

#function to add location features to the data
def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("location", group_keys=False)

    #creating features for the precipitation and discharge lags
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

#function to create labels for the data
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

#function to calculate the fbeta score
def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)

#function to create a mask for the physics gate
def physics_gate_mask(station_train: pd.DataFrame, station_test: pd.DataFrame) -> np.ndarray:
    precip_floor = station_train["precip_sum7"].quantile(PHYSICS_PRECIP7_TRAIN_QUANTILE)
    discharge_floor = station_train["discharge_lag1"].quantile(
        PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE
    )
    heavy_rain = station_test["precip_sum7"].to_numpy() >= precip_floor
    elevated_flow = station_test["discharge_lag1"].to_numpy() >= discharge_floor
    return heavy_rain | elevated_flow

#function to sample the forest parameters
def _sample_forest_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.choice([6, 8, 10, 12, 16])),
        "min_samples_leaf": int(rng.choice([5, 8, 12, 16, 20, 25])),
        "min_samples_split": int(rng.choice([12, 20, 30, 40, 50])),
        "max_features": (
            float(rng.choice([0.45, 0.55]))
            if rng.random() < 0.35
            else rng.choice(["sqrt", "log2"])
        ),
        "class_weight": rng.choice(["balanced", "balanced_subsample"]),
    }

#function to make the random forest model
def _make_rf(params: dict, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )

#function to perform the hyperparameter search
def hyperparameter_search(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    best_score = -1.0
    best_params = _sample_forest_params(rng)

    y_val_np = y_val.to_numpy()
    if len(np.unique(y_val_np)) < 2:
        return best_params

    for _ in range(n_iter):
        params = _sample_forest_params(rng)
        model = _make_rf(params, SEARCH_N_ESTIMATORS)
        model.fit(X_fit, y_fit)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val_np, val_prob)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params

#function to pick the threshold for the model
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

    # Local refine around best threshold
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

#function to evaluate the model on the test set
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
    rf_params = hyperparameter_search(
        X_fit, y_fit, X_val, y_val, HYPERPARAMETER_SEARCH_ITERATIONS
    )
    model = _make_rf(rf_params, FINAL_N_ESTIMATORS)
    model.fit(X_fit, y_fit)

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
        "best_rf_params": rf_params,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "search_seconds": elapsed,
        "roc_auc": roc_auc_score(y_true, test_prob),
        "accuracy": accuracy_score(y_true, test_pred),
        "precision": precision_score(y_true, test_pred, zero_division=0),
        "recall": recall_score(y_true, test_pred, zero_division=0),
        "f1": test_f1,
        "y_true": y_true,
        "y_pred": test_pred,
        "y_prob": test_prob,
    }

#function to flatten the random forest parameters
def _flatten_rf_params(params: dict) -> dict:
    return {f"rf_{key}": value for key, value in params.items()}

#function to write the key values to the file
def _write_key_values(fh, mapping: dict, prefix: str = "") -> None:
    for key, value in mapping.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _write_key_values(fh, value, prefix=f"{label}.")
        else:
            fh.write(f"{label} = {value}\n")

#function to set the pipeline settings
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
        "use_probability_calibration": USE_PROBABILITY_CALIBRATION,
        "use_physics_gate": USE_PHYSICS_GATE,
        "physics_precip7_train_quantile": PHYSICS_PRECIP7_TRAIN_QUANTILE,
        "physics_discharge_lag1_train_quantile": PHYSICS_DISCHARGE_LAG1_TRAIN_QUANTILE,
        "alert_prob_floor": ALERT_PROB_FLOOR,
        "hyperparameter_search_iterations": HYPERPARAMETER_SEARCH_ITERATIONS,
        "search_n_estimators": SEARCH_N_ESTIMATORS,
        "final_n_estimators": FINAL_N_ESTIMATORS,
        "threshold_grid_points": THRESHOLD_GRID_POINTS,
    }

#function to save the run artifacts
def save_run_artifacts(
    fold_rows: list[dict],
    macro: dict,
    micro: dict,
    false_alarms: int,
    alerts: int,
    pooled_true: list[int],
    pooled_pred: list[int],
    pooled_prob: list[float],
    total_wall_seconds: float,
    avg_tune_seconds: float,
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
            "total_wall_seconds": total_wall_seconds,
            "avg_tune_seconds_per_fold": avg_tune_seconds,
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
    settings["total_wall_seconds"] = total_wall_seconds

    best_f1_row = max(fold_rows, key=lambda r: r.get("f1", -1)) if fold_rows else None

    settings_path = RESULTS_DIR / f"{MODEL_TAG}_settings_{run_ts}.txt"
    with settings_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Model run: {MODEL_TAG}\n")
        fh.write(f"Timestamp (UTC): {run_ts}\n\n")
        fh.write("=== Pipeline / search settings ===\n")
        for key, value in settings.items():
            if key in ("summary_macro", "summary_micro"):
                continue
            fh.write(f"{key} = {value}\n")
        fh.write("\n=== Best fold by test F1 (includes tuned RF params) ===\n")
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
        fh.write(f"total_wall_seconds = {total_wall_seconds:.2f}\n")
        fh.write(f"avg_tune_seconds_per_fold = {avg_tune_seconds:.2f}\n")
        fh.write(
            f"\nPer-fold RF params are columns rf_* in {metrics_path.name}\n"
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

#function to main the program
def main() -> None:
    started = time.perf_counter()
    df = add_location_features(load_data())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    split_metrics: list[dict] = []
    fold_rows: list[dict] = []
    scorable_metrics: list[dict] = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_prob: list[float] = []

    print(
        f"Tuned Random Forest (per-station time-series CV)\n"
        f"Rows: {len(df)} | Stations: {df['location'].nunique()} | "
        f"Search iters/fold: {HYPERPARAMETER_SEARCH_ITERATIONS} | "
        f"Final trees: {FINAL_N_ESTIMATORS}\n"
        f"Risk quantile: {RISK_QUANTILE} | Min precision: {MIN_PRECISION} | "
        f"Alert floor: {ALERT_PROB_FLOOR}\n"
    )

    for location in sorted(df["location"].unique()):
        station_df = df[df["location"] == location].sort_values("date")
        station_dates = station_df["date"].unique()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(station_dates)):
            station_train = station_df[station_df["date"].isin(station_dates[train_idx])]
            station_test = station_df[station_df["date"].isin(station_dates[test_idx])]

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
                "search_seconds": result["search_seconds"],
                "train_f1": result["train_f1"],
                "roc_auc": result["roc_auc"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
            fold_row.update(_flatten_rf_params(result["best_rf_params"]))
            fold_rows.append(fold_row)
            failed = result["f1"] == 0 and result["y_true"].sum() >= MIN_TEST_POSITIVES
            if not failed:
                scorable_metrics.append(result)
            pooled_true.extend(result["y_true"].tolist())
            pooled_pred.extend(result["y_pred"].tolist())
            pooled_prob.extend(result["y_prob"].tolist())

            gap = result["train_f1"] - result["test_f1"]
            print(
                f"{location} fold {fold + 1}: thr={result['threshold']:.2f} | "
                f"Prec={result['precision'] * 100:.1f}% | "
                f"Rec={result['recall'] * 100:.1f}% | "
                f"F1={result['f1'] * 100:.1f}% | "
                f"gap={gap * 100:.1f}pp"
            )

    if not split_metrics:
        print("No valid station folds to report.")
        return

    macro_source = scorable_metrics if scorable_metrics else split_metrics
    macro = {
        key: float(np.mean([m[key] for m in macro_source]))
        for key in ("roc_auc", "accuracy", "precision", "recall", "f1", "train_f1", "test_f1")
    }
    avg_tune = float(np.mean([m["search_seconds"] for m in split_metrics]))

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
        f"({len(macro_source)} for macro mean) | "
        f"Avg tuning time/fold: {avg_tune:.1f}s | "
        f"Total wall time: {time.perf_counter() - started:.1f}s\n"
    )
    print("--- Macro mean ---")
    print(
        f"ROC-AUC={macro['roc_auc'] * 100:.2f}% | "
        f"Acc={macro['accuracy'] * 100:.2f}% | "
        f"Prec={macro['precision'] * 100:.2f}% | "
        f"Rec={macro['recall'] * 100:.2f}% | "
        f"F1={macro['f1'] * 100:.2f}%"
    )
    print("--- Micro (pooled test) ---")
    print(
        f"ROC-AUC={micro['roc_auc'] * 100:.2f}% | "
        f"Acc={micro['accuracy'] * 100:.2f}% | "
        f"Prec={micro['precision'] * 100:.2f}% | "
        f"Rec={micro['recall'] * 100:.2f}% | "
        f"F1={micro['f1'] * 100:.2f}%"
    )
    print(
        f"False alarms: {false_alarms} / {alerts} alerts "
        f"({false_alarm_rate * 100:.1f}% wrong)"
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
        time.perf_counter() - started,
        avg_tune,
    )


if __name__ == "__main__":
    main()
