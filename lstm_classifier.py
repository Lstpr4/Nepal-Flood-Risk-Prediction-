"""PyTorch LSTM baseline — same temporal CV and alert logic as XGB/RF."""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset

from pipeline_common import (
    ALERT_PROB_FLOOR,
    FEATURES,
    MIN_PRECISION,
    MIN_TEST_POSITIVES,
    MIN_TRAIN_POSITIVES,
    MIN_TRAIN_ROWS,
    N_SPLITS,
    PREDICTION_HORIZON_DAYS,
    RANDOM_STATE,
    RESULTS_DIR,
    RISK_QUANTILE,
    USE_PHYSICS_GATE,
    USE_PROBABILITY_CALIBRATION,
    VAL_FRACTION,
    add_location_features,
    load_data,
    physics_gate_mask,
    pick_threshold,
    prepare_station_horizon_frames,
    split_station_train_val,
)

MODEL_TAG = "lstm"
SEQUENCE_LENGTH = 14
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.25
MAX_EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RiverLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _history_window(station_df: pd.DataFrame, end_date: pd.Timestamp) -> np.ndarray:
    hist = station_df[station_df["date"] <= end_date].tail(SEQUENCE_LENGTH)
    arr = hist[FEATURES].to_numpy(dtype=np.float32)
    if len(arr) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(arr), len(FEATURES)), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr


def frames_to_sequences(station_full: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    seqs = []
    for end_date in frame["date"]:
        seqs.append(_history_window(station_full, end_date))
    return np.stack(seqs, axis=0)


def _train_lstm(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RiverLSTM:
    torch.manual_seed(RANDOM_STATE)
    pos = max(int(y_fit.sum()), 1)
    neg = len(y_fit) - pos
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = RiverLSTM(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit),
            torch.from_numpy(y_fit.astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_state = None
    best_auc = -1.0
    stale = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(DEVICE))
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        if len(np.unique(y_val)) >= 2:
            val_auc = roc_auc_score(y_val, val_prob)
        else:
            val_auc = 0.0

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_proba(model: RiverLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        return torch.sigmoid(logits).cpu().numpy()


def evaluate_station_split(
    station_full: pd.DataFrame,
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

    tr = tr.sort_values("date")
    te = te.sort_values("date")
    y_train = y_train.loc[tr.index]
    y_test = y_test.loc[te.index]

    unique_dates = np.sort(tr["date"].unique())
    n_val = max(20, int(len(unique_dates) * VAL_FRACTION))
    val_dates = set(unique_dates[-n_val:])
    fit_mask = ~tr["date"].isin(val_dates)
    val_mask = tr["date"].isin(val_dates)

    X_tr = frames_to_sequences(station_full, tr)
    y_tr = y_train.to_numpy()
    fit_np = fit_mask.to_numpy()
    val_np = val_mask.to_numpy()
    X_fit = X_tr[fit_np]
    y_fit = y_tr[fit_np]
    X_val = X_tr[val_np]
    y_val = y_tr[val_np]
    X_test = frames_to_sequences(station_full, te)
    y_true = y_test.to_numpy()

    if y_fit.sum() == 0 or len(np.unique(y_val)) < 2:
        return None

    t0 = time.perf_counter()
    model = _train_lstm(X_fit, y_fit, X_val, y_val)
    fit_prob = _predict_proba(model, X_fit)
    val_prob = _predict_proba(model, X_val)
    test_prob = _predict_proba(model, X_test)

    if USE_PROBABILITY_CALIBRATION and len(np.unique(y_val)) >= 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_prob, y_val)
        fit_prob = calibrator.predict(fit_prob)
        val_prob = calibrator.predict(val_prob)
        test_prob = calibrator.predict(test_prob)

    threshold = pick_threshold(y_val, val_prob, min_precision)
    effective_threshold = max(threshold, ALERT_PROB_FLOOR)
    train_f1 = f1_score(y_fit, (fit_prob >= effective_threshold).astype(int), zero_division=0)
    test_pred = (test_prob >= effective_threshold).astype(int)

    if USE_PHYSICS_GATE:
        gate = physics_gate_mask(station_train, te)
        test_pred = (test_pred.astype(bool) & gate).astype(int)

    elapsed = time.perf_counter() - t0
    test_f1 = f1_score(y_true, test_pred, zero_division=0)
    print(f"  [{location} fold {fold}] LSTM trained in {elapsed:.1f}s", flush=True)

    return {
        "threshold": effective_threshold,
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

    settings_path = RESULTS_DIR / f"{MODEL_TAG}_settings_{run_ts}.txt"
    with settings_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Model run: {MODEL_TAG}\n")
        fh.write(f"Timestamp (UTC): {run_ts}\n")
        fh.write(f"sequence_length = {SEQUENCE_LENGTH}\n")
        fh.write(f"hidden_size = {HIDDEN_SIZE}\n")
        fh.write(f"prediction_horizon_days = {PREDICTION_HORIZON_DAYS}\n")
        fh.write(f"macro = {macro}\n")
        fh.write(f"micro = {micro}\n")
        fh.write(f"false_alarms = {false_alarms}\n")
        fh.write(f"alerts = {alerts}\n")

    latest_metrics = RESULTS_DIR / f"{MODEL_TAG}_metrics_latest.csv"
    latest_settings = RESULTS_DIR / f"{MODEL_TAG}_settings_latest.txt"
    metrics_df.to_csv(latest_metrics, index=False)
    latest_settings.write_text(settings_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\nSaved LSTM results to {RESULTS_DIR}/")
    print(f"  {metrics_path.name}")


def main() -> None:
    df = add_location_features(load_data())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    split_metrics: list[dict] = []
    fold_rows: list[dict] = []
    scorable_metrics: list[dict] = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_prob: list[float] = []

    print(
        f"PyTorch LSTM | Device: {DEVICE} | Seq len: {SEQUENCE_LENGTH} | "
        f"Horizon: {PREDICTION_HORIZON_DAYS} day(s)\n"
    )

    for location in sorted(df["location"].unique()):
        station_df = df[df["location"] == location].sort_values("date")
        station_dates = station_df["date"].unique()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(station_dates)):
            station_train = station_df[station_df["date"].isin(station_dates[train_idx])]
            station_test = station_df[station_df["date"].isin(station_dates[test_idx])]

            result = evaluate_station_split(
                station_df,
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
            fold_rows.append(
                {
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
            )
            failed = result["f1"] == 0 and result["y_true"].sum() >= MIN_TEST_POSITIVES
            if not failed:
                scorable_metrics.append(result)
            pooled_true.extend(result["y_true"].tolist())
            pooled_pred.extend(result["y_pred"].tolist())
            pooled_prob.extend(result["y_prob"].tolist())

            print(
                f"{location} fold {fold + 1}: F1={result['f1'] * 100:.1f}% | "
                f"Prec={result['precision'] * 100:.1f}% | Rec={result['recall'] * 100:.1f}%"
            )

    if not split_metrics:
        print("No valid LSTM folds.")
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

    print(
        f"\nLSTM macro F1={macro['f1'] * 100:.2f}% | "
        f"micro F1={micro['f1'] * 100:.2f}% | "
        f"false alarms={false_alarms}/{alerts}"
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
