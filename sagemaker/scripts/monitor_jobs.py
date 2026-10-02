#!/usr/bin/env python3
"""Poll the 4 training jobs to completion; save full logs on terminal state."""
import os
import subprocess
import time

JOBS = [
    "goemotions-sft-2026-10-02-16-07-29-859",
    "goemotions-rl-2026-10-02-16-30-25-068",
    "orena-sft-2026-10-02-17-58-46-825",
    "orena-rl-2026-10-02-17-18-25-972",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(OUT, exist_ok=True)
TERMINAL = {"Completed", "Failed", "Stopped"}


def describe(job):
    r = subprocess.run(
        ["aws", "sagemaker", "describe-training-job", "--training-job-name", job,
         "--query", "TrainingJobStatus", "--output", "text"],
        capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def save_log(job):
    path = os.path.join(OUT, f"{job}.log")
    with open(path, "w") as f:
        subprocess.run(
            ["aws", "logs", "tail", "/aws/sagemaker/TrainingJobs",
             "--log-stream-name-prefix", job, "--format", "short"],
            stdout=f, stderr=subprocess.DEVNULL)
    n = sum(1 for _ in open(path, encoding="utf-8", errors="ignore"))
    print(f"TERMINAL [{job}] -> {path} ({n} lines)", flush=True)


unfinished = list(JOBS)
while unfinished:
    still = []
    for j in unfinished:
        st = describe(j)
        if st in TERMINAL:
            print(f"TERMINAL [{j}] status={st}", flush=True)
            save_log(j)
        else:
            still.append(j)
    unfinished = still
    if unfinished:
        time.sleep(60)

print("ALL_TERMINAL", flush=True)
