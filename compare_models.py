"""Build three-way comparison table from latest metrics CSVs."""

from pathlib import Path

import pandas as pd

from pipeline_common import RESULTS_DIR

MODELS = [
    ("XGBoost", "xgb"),
    ("Random Forest", "random_forest"),
    ("PyTorch LSTM", "lstm"),
]

OUT_CSV = RESULTS_DIR / "model_comparison_latest.csv"
OUT_MD = RESULTS_DIR / "model_comparison_latest.md"


def _read_summary(tag: str) -> dict | None:
    path = RESULTS_DIR / f"{tag}_metrics_latest.csv"
    if not path.is_file():
        print(f"Skip {tag}: missing {path.name}")
        return None
    df = pd.read_csv(path)
    macro = df[df["level"] == "macro"]
    micro = df[df["level"] == "micro"]
    op = df[df["level"] == "operational"]
    if macro.empty or micro.empty:
        return None
    row = {"model_tag": tag}
    for prefix, part in [("macro", macro), ("micro", micro)]:
        for col in ("roc_auc", "accuracy", "precision", "recall", "f1"):
            if col in part.columns:
                row[f"{prefix}_{col}"] = float(part.iloc[0][col])
    if not op.empty:
        row["false_alarms"] = op.iloc[0].get("false_alarms")
        row["alerts"] = op.iloc[0].get("alerts")
        row["false_alarm_rate"] = op.iloc[0].get("false_alarm_rate")
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for display_name, tag in MODELS:
        summary = _read_summary(tag)
        if summary is None:
            continue
        summary["model"] = display_name
        rows.append(summary)

    if not rows:
        raise SystemExit("No metrics files found. Run training scripts first.")

    df = pd.DataFrame(rows)
    cols = [
        "model",
        "macro_roc_auc",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "micro_roc_auc",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "false_alarm_rate",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    pct_cols = [c for c in df.columns if c not in ("model", "false_alarm_rate")]
    display = df.copy()
    for c in pct_cols:
        if c == "false_alarm_rate":
            display[c] = (display[c].astype(float) * 100).round(1).astype(str) + "%"
        elif c != "model":
            display[c] = (display[c].astype(float) * 100).round(1).astype(str) + "%"

    lines = [
        "# Model comparison (latest runs)",
        "",
        "Same temporal CV, horizon, and alert settings per script.",
        "",
        display.to_markdown(index=False),
        "",
        f"CSV: `{OUT_CSV.name}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(display.to_string(index=False))
    print(f"\nWrote {OUT_CSV.name} and {OUT_MD.name}")


if __name__ == "__main__":
    main()
