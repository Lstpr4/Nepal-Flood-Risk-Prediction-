"""FastAPI inference for per-station XGBoost discharge risk scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline_common import ARTIFACTS_DIR, FEATURES

BUNDLE_PATH = ARTIFACTS_DIR / "serving_bundle.joblib"

app = FastAPI(
    title="Nepal River Discharge Risk API",
    description="Scores high discharge risk from daily feature vectors (XGBoost serving bundle).",
    version="1.0.0",
)

_bundle: dict[str, Any] | None = None


def _load_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not BUNDLE_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {BUNDLE_PATH}. Run: python export_serving_model.py"
            )
        _bundle = joblib.load(BUNDLE_PATH)
    return _bundle


class PredictRequest(BaseModel):
    location: str = Field(..., examples=["Devghat"])
    features: dict[str, float] = Field(
        ...,
        description=f"Keys must include all model features ({len(FEATURES)} fields).",
    )


class PredictResponse(BaseModel):
    location: str
    probability: float
    alert: bool
    threshold: float
    physics_gate_pass: bool


@app.on_event("startup")
def startup() -> None:
    try:
        _load_bundle()
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")


@app.get("/health")
def health() -> dict[str, str | int | bool]:
    ok = BUNDLE_PATH.is_file()
    n_loc = 0
    if ok:
        n_loc = len(_load_bundle().get("locations", {}))
    return {"status": "ok" if ok else "no_model", "locations_loaded": n_loc}


@app.get("/features")
def list_features() -> dict[str, list[str]]:
    bundle = _load_bundle()
    return {"features": bundle.get("features", FEATURES)}


@app.get("/locations")
def list_locations() -> dict[str, list[str]]:
    bundle = _load_bundle()
    return {"locations": sorted(bundle.get("locations", {}).keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    bundle = _load_bundle()
    locs = bundle["locations"]
    if body.location not in locs:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown location. Choose from: {sorted(locs.keys())}",
        )

    feat_names = bundle["features"]
    missing = [f for f in feat_names if f not in body.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}",
        )

    row = np.array([[float(body.features[f]) for f in feat_names]], dtype=np.float32)
    entry = locs[body.location]
    model = entry["model"]
    prob = float(model.predict_proba(row)[0, 1])
    calibrator = entry.get("calibrator")
    if calibrator is not None:
        prob = float(calibrator.predict([prob])[0])

    threshold = float(entry["threshold"])
    physics_pass = True
    if bundle.get("use_physics_gate") and entry.get("physics"):
        p = entry["physics"]
        precip7 = float(body.features.get("precip_sum7", 0))
        lag1 = float(body.features.get("discharge_lag1", 0))
        physics_pass = (
            precip7 >= p["precip_sum7_floor"] or lag1 >= p["discharge_lag1_floor"]
        )

    alert = prob >= threshold and physics_pass

    return PredictResponse(
        location=body.location,
        probability=prob,
        alert=alert,
        threshold=threshold,
        physics_gate_pass=physics_pass,
    )
