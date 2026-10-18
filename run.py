# subprocess used to execute commands from python
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

TRAIN_PROGRAMS = [
    "XGBClassifier.py",
    "random_forest.py",
    "lstm_classifier.py",
]

POST_TRAIN = [
    "export_serving_model.py",
    "compare_models.py",
]


def _run_parallel(scripts: list[str]) -> list[tuple[str, int]]:
    running: list[tuple[str, subprocess.Popen]] = []
    for script in scripts:
        path = BASE_DIR / script
        print(f"Starting: {path.name}")
        proc = subprocess.Popen([sys.executable, str(path)], cwd=BASE_DIR)
        running.append((script, proc))

    outcomes: list[tuple[str, int]] = []
    for script, proc in running:
        code = proc.wait()
        outcomes.append((script, code))
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"Finished: {script} — {status}")
    return outcomes


def _run_sequential(scripts: list[str]) -> list[tuple[str, int]]:
    outcomes: list[tuple[str, int]] = []
    for script in scripts:
        path = BASE_DIR / script
        print(f"Running: {path.name}")
        code = subprocess.call([sys.executable, str(path)], cwd=BASE_DIR)
        outcomes.append((script, code))
        if code != 0:
            print(f"Stopped after {script} failed.")
            break
    return outcomes


def main() -> None:
    print(f"\n=== Phase 1: train {len(TRAIN_PROGRAMS)} models in parallel ===\n")
    train_out = _run_parallel(TRAIN_PROGRAMS)
    failed_train = [s for s, c in train_out if c != 0]
    if failed_train:
        raise SystemExit(f"Training failed: {', '.join(failed_train)}")

    print(f"\n=== Phase 2: export API bundle + comparison table ===\n")
    post_out = _run_sequential(POST_TRAIN)
    failed_post = [s for s, c in post_out if c != 0]
    if failed_post:
        raise SystemExit(f"Post-train step failed: {', '.join(failed_post)}")

    print("\nAll steps completed.")
    print("  results/model_comparison_latest.md")
    print("  artifacts/serving_bundle.joblib")
    print("  API: uvicorn serve:app --reload")


if __name__ == "__main__":
    main()
