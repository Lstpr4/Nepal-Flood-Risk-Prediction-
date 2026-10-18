# Nepal River Discharge & Flood-Risk Prediction

Machine learning pipelines to flag **elevated river discharge risk** across **10 monitoring stations** in Nepal, using daily weather, soil moisture, and hydrology data. The goal is not “winning on accuracy alone,” but building **honest, time-aware models** that balance **early warning** with **trustworthy alerts**—because when rivers rise, **hours and minutes decide who gets out in time**.

---

## Why this project exists

Nepal sits at the collision of the Himalayas, monsoon extremes, and fast-changing river systems. Floods and sudden discharge spikes do not announce themselves on a calendar; they arrive through rain on saturated slopes, melting snow, and channels that can go from normal to dangerous in a single day. For communities along those rivers, **better prediction is not a nice-to-have—it is a lifeline**. Warnings that come too late cost lives; warnings that cry wolf every week erode trust until the real event is ignored.

I started this project because **existing approaches often fail in the ways that matter most in the real world**: models trained as if the past and future were interchangeable, metrics that look strong on paper but hide false alarms, and single global thresholds that ignore how different each river behaves. I wanted a reproducible baseline that treats flood risk as a **per-station, time-series problem**—with explicit trade-offs between **precision, recall, and false alarms**—so that improvements could eventually feed **faster, more credible early warning**, not just another leaderboard score.

---

## Dataset

A **daily, multi-location** dataset combining:

- Weather (precipitation, temperature, humidity, wind, etc.)
- **Soil moisture**
- **River discharge** (`river_discharge_m3s`)

**~13,390 rows** across **10 strategically selected stations**, spanning **Himalayan, mid-hill, and downstream** river systems. Together, these locations form a practical cross-section of how atmospheric and land-surface conditions relate to river behaviour across Nepal—useful for hydrology, environmental analysis, time-series modelling, and ML.

**Source:** [Nepal Flood and Weather Dataset (2023–2026)](https://www.kaggle.com/datasets/tejal5kunjir/nepal-flood-and-weather-dataset-2023-2026) on Kaggle (`CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv`).

---

### Situation

River flooding in Nepal is driven by complex, localized signals: monsoon bursts, antecedent wetness, elevation, and station-specific channel response. Disaster response depends on **detecting abnormal discharge early enough** for evacuation, dam operations, and resource deployment—yet many ML demos use random splits, optimize accuracy, and never measure **how often the model cries wolf**. That gap between “model works in a notebook” and “model could help humans escape” is exactly what motivated this work.

### Task

Build a **rigorous, comparable** flood-/discharge-risk modelling pipeline that:

1. Respects **time** (no leakage from the future into the past).
2. Evaluates **each station** on its own hydrological behaviour.
3. Optimizes for **operational metrics**—F1, precision, recall, and **false alarm rate**—not accuracy alone.
4. Supports **next-day (1-day-ahead) risk** so predictions align with **lead time** for human response.
5. Compares **XGBoost**, **Random Forest**, and **PyTorch LSTM** under the same evaluation contract.
6. **Docker** reproducibility, **FastAPI** inference, and an auto-generated **three-way metrics table**.

### Action

- **Feature engineering:** per-station lags and rolling sums for precipitation and discharge, saturation interactions, and discharge-vs-baseline ratios.
- **Labels:** high-risk days defined by a **train-only** discharge quantile (default **85th percentile**), with optional **1-day forecast horizon** (predict tomorrow’s high discharge from today’s features).
- **Validation:** **per-station `TimeSeriesSplit`** (3 folds), date-based train/validation holdout, and fold-level **hyperparameter search** (XGB: random search on validation AUC; RF: analogous forest search).
- **Deployment-minded post-processing:**
  - **Isotonic probability calibration**
  - **Threshold tuning** (F0.5 with minimum precision/recall constraints)
  - **Alert probability floor** to suppress low-confidence alarms
  - Optional **physics gate** (recent 7-day rain or elevated lagged discharge) so alerts align with plausible flood drivers
- **Reproducible outputs:** metrics, pooled predictions, and plain-text settings under `results/`.
- **`run.py`:** trains **XGBoost, Random Forest, and LSTM in parallel**, exports serving models, builds comparison table.
- **Docker / FastAPI:** containerized runs + REST `/predict` for per-station risk scores.
- **`compare_models.py`:** writes `results/model_comparison_latest.md` (three-way table).

### Result

On the latest XGBoost run (**1-day horizon**, search enabled), pooled evaluation shows the model is **strong at ranking risk** and **moderately conservative on alerts**:

| Level | ROC-AUC | Precision | Recall | F1 |
|--------|---------|-----------|--------|-----|
| **Macro** (mean over station-folds) | ~95.5% | ~80.2% | ~72.8% | ~72.8% |
| **Micro** (all pooled test rows) | ~93.3% | ~83.6% | ~54.2% | ~65.7% |

**Operational:** ~**980** alerts on held-out test rows, with ~**16.4%** false-alarm rate among those alerts (~161 false positives)—a deliberate trade-off toward **trustworthy warnings** rather than firing on every uncertain spike.

*(Exact numbers vary by run; see `results/xgb_metrics_latest.csv` and `results/xgb_settings_latest.txt`.)*

---

## Impact on today’s world

This project is a **research and engineering baseline**, not a deployed national EWS—but the design choices mirror what real warning systems need:

1. **Lead time:** 1-day-ahead targets connect ML outputs to **“tomorrow’s risk”**, the window where SMS alerts, sirens, and evacuation planning actually happen.
2. **False alarms matter:** In disaster settings, **each bogus alert trains people to ignore the next one**. Tracking false alarms alongside F1 makes improvement measurable in **human terms** (trust, compliance, lives).
3. **Local rivers, local thresholds:** A single Nepal-wide rule would fail; **per-station CV and thresholds** respect that Koshi, Karnali, and mid-hill tributaries do not share one “normal” discharge.
4. **Comparable models:** **XGBoost vs Random Forest vs LSTM** under the same splits supports evidence-based choices before deployment.
5. **Docker + API:** Shows MLOps-style packaging—not just notebooks.

If integrated with **IMD/monsoon feeds, river gauges, and municipal alerting**, pipelines like this could **shorten the path from data to decision**: prioritizing villages, staging rescue assets, and giving families **one more night** to move uphill when the model and physics agree the river is trending wrong.

---

## Future directions

Ways this work could be ** pushed further** toward production-grade early warning:

| Direction | Why it matters |
|-----------|----------------|
| **Longer horizons (2–7 days)** | Align with weather forecast skill and regional flood response plans. |
| **Ensemble + uncertainty** | Conformal intervals or quantile models to say *“80% chance of high discharge”* not just binary alerts. |
| **Real-time ingestion** | Stream gauges and radar-QPE instead of batch CSV; latency budgets in minutes, not days. |
| **Spatial modelling** | Graph or basin-wide models linking upstream → downstream stations for cascade floods. |
| **Human-in-the-loop evaluation** | Case studies on known 2023–2026 events; expert review of missed hits and false alarms. |
| **Fairness & coverage** | Explicit metrics for remote stations with sparse data so warnings do not only work where sensors are richest. |
| **Deployment** | ✅ FastAPI starter done; add dashboard, cell-broadcast hooks, drills with authorities. |
| **Physics + ML hybrid** | Rainfall–runoff simulators as features or sanity checks alongside learned models. |

---

## Project structure

```
├── pipeline_common.py      # shared data, labels, thresholds
├── XGBClassifier.py          # XGBoost + HPO
├── random_forest.py          # Random Forest + HPO (1-day horizon aligned)
├── lstm_classifier.py        # PyTorch LSTM baseline (14-day sequences)
├── export_serving_model.py   # per-station XGB bundle for API
├── serve.py                  # FastAPI inference
├── compare_models.py         # three-way metrics table
├── run.py                    # full pipeline orchestration
├── Dockerfile / docker-compose.yml
├── artifacts/                # serving_bundle.joblib (generated)
└── results/                  # metrics, model_comparison_latest.md
```

---

## Three-way model comparison

After `python run.py`, open **`results/model_comparison_latest.md`** (or `.csv`).  
Same temporal CV and 1-day horizon; LSTM uses 14-day feature sequences.

---

## Requirements

```bash
pip install -r requirements.txt
```

Configure [Kaggle API credentials](https://github.com/Kaggle/kaggle-api) for `kagglehub`.

---

## How to run (local)

**Full pipeline** (train 3 models → export API bundle → comparison table):

```bash
python run.py
```

**Single components:**

```bash
python XGBClassifier.py
python random_forest.py
python lstm_classifier.py
python export_serving_model.py
python compare_models.py
```

**REST API** (after export):

```bash
python export_serving_model.py   # if artifacts/serving_bundle.joblib is missing
uvicorn serve:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs** (use `127.0.0.1`, not `0.0.0.0`).

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Bundle loaded? (`status: ok` means ready) |
| `GET /locations` | Station names |
| `GET /locations/Devghat/sample` | Full example `features` dict for one station |
| `POST /predict/demo` | Body: `{"location": "Devghat"}` — easiest test (uses bundled sample row) |
| `POST /predict` | Body: `{"location": "...", "features": { all 27 fields }}` |

If `/health` shows `no_model`, run `python export_serving_model.py` first. After re-export, restart uvicorn.

---

## Docker

Train everything inside a container (mounts `results/` and `artifacts/`):

```bash
docker compose build
docker compose run --rm train
```

Export + API (profiles):

```bash
docker compose --profile export run --rm export
docker compose --profile api up api
```

Copy `.env.example` → `.env` and add your Kaggle username and API key (from [Kaggle Settings → API](https://www.kaggle.com/settings)). Docker Compose reads `.env` automatically for dataset download.

---

## Key settings

Shared in `pipeline_common.py` (and XGB-specific toggles in `XGBClassifier.py`):

- `PREDICTION_HORIZON_DAYS` — `0` same-day, `1` next-day (default **1**)
- `USE_PHYSICS_GATE`, `ALERT_PROB_FLOOR`, `MIN_PRECISION` — alert behaviour

---

## Limitations (honest)

- **Historical daily data only**—not a live operational system.  
- **Proxy labels** (high discharge vs station quantile), not official flood inundation maps.  
- **No demographic or infrastructure exposure** layer (who lives in the path).  
- Metrics are **CV on 2023–2026**; external validation on held-out monsoon seasons or events is the next bar before claiming lives saved.

---

## Author

**Jesus** — software engineer; built this as a portfolio-grade hydrology ML project focused on **time-series rigour** and **alert quality** for Nepal’s river systems.

---

*Every improvement in lead time and false-alarm control is a step toward a world where rivers still rise, but fewer people are surprised when they do.*
