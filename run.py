# subprocess used to execute commands from python
import subprocess
import sys
# library that allows easy access to file paths
from pathlib import Path

# project root (folder containing this file)
BASE_DIR = Path(__file__).resolve().parent


# all training/evaluation scripts to run
programs = [
    "XGBClassifier.py",
    "random_forest.py",
]

running: list[tuple[str, subprocess.Popen]] = []

for program in programs:
    program_path = BASE_DIR / program
    print(f"Starting program: {program_path}")
    proc = subprocess.Popen(
        [sys.executable, str(program_path)],
        cwd=BASE_DIR,
    )
    running.append((program, proc))

print(f"\nRunning {len(running)} models in parallel…\n")

failures: list[tuple[str, int]] = []
for program, proc in running:
    return_code = proc.wait()
    if return_code != 0:
        failures.append((program, return_code))
        print(f"Failed: {program} (exit code {return_code})")
    else:
        print(f"Finished: {program}")

if failures:
    details = ", ".join(f"{name}={code}" for name, code in failures)
    raise SystemExit(f"One or more programs failed: {details}")

print("\nAll models completed successfully.")
