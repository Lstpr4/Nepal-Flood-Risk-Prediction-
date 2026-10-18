"""FastAPI inference for per-station XGBoost discharge risk scores."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from pipeline_common import ARTIFACTS_DIR, FEATURES

BUNDLE_PATH = ARTIFACTS_DIR / "serving_bundle.joblib"

_bundle: dict[str, Any] | None = None
_bundle_error: str | None = None


def _load_bundle(force: bool = False) -> dict[str, Any]:
    global _bundle, _bundle_error
    if _bundle is not None and not force:
        return _bundle
    if not BUNDLE_PATH.is_file():
        _bundle_error = (
            f"Model bundle not found at {BUNDLE_PATH}. "
            "Run: python export_serving_model.py"
        )
        raise FileNotFoundError(_bundle_error)
    try:
        _bundle = joblib.load(BUNDLE_PATH)
        _bundle_error = None
        return _bundle
    except Exception as exc:  # noqa: BLE001 — surface load errors to API clients
        _bundle_error = f"Failed to load bundle: {exc}"
        raise RuntimeError(_bundle_error) from exc


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _load_bundle()
        print(f"Loaded serving bundle from {BUNDLE_PATH}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"WARNING: {exc}")
    yield


app = FastAPI(
    title="Nepal River Discharge Risk API",
    description=(
        "Scores high discharge risk from daily feature vectors (XGBoost). "
        "Use POST /predict/demo with only a station name to try a bundled sample row, "
        "or POST /predict with the full feature dict from GET /locations/{name}/sample."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "location": "Devghat",
                    "features": {"precipitation_mm": 12.5, "discharge_lag1": 850.0},
                }
            ]
        }
    )

    location: str = Field(..., description="Exact station name from GET /locations")
    features: dict[str, float] = Field(
        ...,
        description=f"All {len(FEATURES)} model inputs (see GET /locations/{{name}}/sample).",
    )


class DemoPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"location": "Devghat"}]}
    )

    location: str = Field(
        ...,
        description="Exact station name (copy from GET /locations, e.g. Devghat)",
        examples=["Devghat"],
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "location": "Devghat",
                    "probability": 0.12,
                    "alert": False,
                    "threshold": 0.68,
                    "physics_gate_pass": True,
                    "demo": True,
                    "sample_date": "2026-08-31",
                }
            ]
        }
    )

    location: str
    probability: float = Field(..., description="Calibrated risk score 0–1")
    alert: bool = Field(..., description="True if prob ≥ threshold and physics gate passes")
    threshold: float
    physics_gate_pass: bool
    demo: bool = Field(False, description="True when using POST /predict/demo sample row")
    sample_date: str | None = Field(None, description="Date of bundled sample (demo only)")


def _require_bundle() -> dict[str, Any]:
    try:
        return _load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _run_predict(
    location: str,
    features: dict[str, float],
    *,
    demo: bool = False,
    sample_date: str | None = None,
) -> PredictResponse:
    bundle = _require_bundle()
    locs = bundle["locations"]
    if location not in locs:
        hint = (
            "Swagger often sends the placeholder \"string\" — replace it with a real "
            "station name from GET /locations (e.g. Devghat)."
        )
        if location == "string":
            hint = (
                'You sent location "string" (Swagger default). Use a real name like '
                '"Devghat" from GET /locations.'
            )
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown location",
                "location": location,
                "hint": hint,
                "available": sorted(locs.keys()),
            },
        )

    feat_names: list[str] = list(bundle["features"])
    missing = [f for f in feat_names if f not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features",
                "missing": missing,
                "hint": f"GET /locations/{location}/sample for a full example payload",
            },
        )

    row_df = pd.DataFrame(
        [{name: float(features[name]) for name in feat_names}],
        columns=feat_names,
    )
    entry = locs[location]
    model = entry["model"]
    prob = float(model.predict_proba(row_df)[0, 1])
    calibrator = entry.get("calibrator")
    if calibrator is not None:
        prob = float(calibrator.predict(np.array([prob]))[0])

    threshold = float(entry["threshold"])
    physics_pass = True
    if bundle.get("use_physics_gate") and entry.get("physics"):
        p = entry["physics"]
        precip7 = float(features.get("precip_sum7", 0))
        lag1 = float(features.get("discharge_lag1", 0))
        physics_pass = (
            precip7 >= p["precip_sum7_floor"] or lag1 >= p["discharge_lag1_floor"]
        )

    alert = bool(prob >= threshold and physics_pass)

    return PredictResponse(
        location=location,
        probability=prob,
        alert=alert,
        threshold=threshold,
        physics_gate_pass=physics_pass,
        demo=demo,
        sample_date=sample_date,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Nepal River Discharge Risk API",
        "docs": "/docs",
        "health": "/health",
        "try_demo": "POST /predict/demo with body {\"location\": \"Devghat\"}",
    }


@app.get("/health")
def health() -> dict[str, str | int | bool]:
    if not BUNDLE_PATH.is_file():
        return {
            "status": "no_model",
            "locations_loaded": 0,
            "bundle_path": str(BUNDLE_PATH),
            "fix": "python export_serving_model.py",
        }
    try:
        bundle = _load_bundle()
        return {
            "status": "ok",
            "locations_loaded": len(bundle.get("locations", {})),
            "bundle_path": str(BUNDLE_PATH),
            "prediction_horizon_days": bundle.get("prediction_horizon_days"),
        }
    except RuntimeError as exc:
        return {
            "status": "load_error",
            "locations_loaded": 0,
            "detail": str(exc),
            "fix": "Re-run python export_serving_model.py",
        }


@app.get("/features")
def list_features() -> dict[str, list[str]]:
    bundle = _require_bundle()
    return {"features": bundle.get("features", FEATURES)}


@app.get("/locations")
def list_locations() -> dict[str, list[str]]:
    bundle = _require_bundle()
    return {"locations": sorted(bundle["locations"].keys())}


@app.get("/locations/{location}/sample")
def location_sample(location: str) -> dict:
    bundle = _require_bundle()
    locs = bundle["locations"]
    if location not in locs:
        raise HTTPException(
            status_code=404,
            detail={"available": sorted(locs.keys())},
        )
    entry = locs[location]
    sample = entry.get("sample_features")
    if not sample:
        raise HTTPException(
            status_code=503,
            detail="Bundle has no sample row. Re-run: python export_serving_model.py",
        )
    return {
        "location": location,
        "sample_date": entry.get("sample_date"),
        "features": sample,
        "usage": {
            "predict": "POST /predict with this location and features object",
            "demo": "POST /predict/demo with only location (uses this sample)",
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    return _run_predict(body.location, body.features, demo=False)


@app.post("/predict/demo", response_model=PredictResponse)
def predict_demo(body: DemoPredictRequest) -> PredictResponse:
    bundle = _require_bundle()
    entry = bundle["locations"].get(body.location)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"available": sorted(bundle["locations"].keys())},
        )
    sample = entry.get("sample_features")
    if not sample:
        raise HTTPException(
            status_code=503,
            detail="Re-export models: python export_serving_model.py",
        )
    return _run_predict(
        body.location,
        sample,
        demo=True,
        sample_date=entry.get("sample_date"),
    )
