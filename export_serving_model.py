"""Train per-station XGBoost models and save a bundle for FastAPI inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from pipeline_common import (
    ALERT_PROB_FLOOR,
    ARTIFACTS_DIR,
    FEATURES,
    MIN_PRECISION,
    PREDICTION_HORIZON_DAYS,
    RANDOM_STATE,
    RISK_QUANTILE,
    USE_PHYSICS_GATE,
    VAL_FRACTION,
    add_location_features,
    load_data,
    pick_threshold,
    prepare_station_horizon_frames,
    split_station_train_val,
)

BUNDLE_PATH = ARTIFACTS_DIR / "serving_bundle.joblib"
META_PATH = ARTIFACTS_DIR / "serving_bundle_meta.json"


def _scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    return neg / max(pos, 1)


def train_location_bundle(station_df: pd.DataFrame, location: str) -> dict:
    station_df = station_df.sort_values("date")
    n_dates = len(station_df["date"].unique())
    if n_dates < 200:
        return {}

    split_at = int(n_dates * (1 - VAL_FRACTION))
    dates = np.sort(station_df["date"].unique())
    train_dates = set(dates[:split_at])
    station_train = station_df[station_df["date"].isin(train_dates)]
    holdout = station_df[~station_df["date"].isin(train_dates)]
    if len(holdout) == 0:
        holdout = station_train.iloc[-30:]
        station_train = station_train.iloc[:-30]

    tr, _, y_train, _ = prepare_station_horizon_frames(
        station_train,
        station_train.iloc[-1:],
        RISK_QUANTILE,
        PREDICTION_HORIZON_DAYS,
    )
    if len(tr) < 100 or y_train.sum() < 3:
        return {}

    X_fit, y_fit, X_val, y_val = split_station_train_val(tr, y_train, VAL_FRACTION)
    if y_fit.sum() == 0 or len(np.unique(y_val)) < 2:
        return {}

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=4,
        subsample=0.88,
        colsample_bytree=0.88,
        scale_pos_weight=_scale_pos_weight(y_fit),
        random_state=RANDOM_STATE,
        eval_metric="auc",
        early_stopping_rounds=40,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    val_prob = model.predict_proba(X_val)[:, 1]
    calibrator = None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_prob, y_val.to_numpy())
    val_prob = calibrator.predict(val_prob)

    threshold = max(pick_threshold(y_val.to_numpy(), val_prob, MIN_PRECISION), ALERT_PROB_FLOOR)

    physics = {}
    if USE_PHYSICS_GATE:
        physics = {
            "precip_sum7_floor": float(
                station_train["precip_sum7"].quantile(0.60)
            ),
            "discharge_lag1_floor": float(
                station_train["discharge_lag1"].quantile(0.65)
            ),
        }

    return {
        "location": location,
        "model": model,
        "calibrator": calibrator,
        "threshold": float(threshold),
        "physics": physics,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_location_features(load_data())

    locations: dict[str, dict] = {}
    for location in sorted(df["location"].unique()):
        print(f"Training serving model for {location}…")
        entry = train_location_bundle(df[df["location"] == location], location)
        if entry:
            locations[location] = entry

    if not locations:
        raise SystemExit("No location models trained; check data and thresholds.")

    bundle = {
        "version": 1,
        "features": FEATURES,
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "alert_prob_floor": ALERT_PROB_FLOOR,
        "use_physics_gate": USE_PHYSICS_GATE,
        "locations": locations,
    }
    joblib.dump(bundle, BUNDLE_PATH)

    meta = {
        "locations": list(locations.keys()),
        "n_features": len(FEATURES),
        "bundle_path": str(BUNDLE_PATH.name),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved {len(locations)} station models to {BUNDLE_PATH}")
    print(f"Metadata: {META_PATH}")


if __name__ == "__main__":
    main()
