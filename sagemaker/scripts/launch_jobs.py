#!/usr/bin/env python3
"""Submit the 4 short SageMaker training jobs (script mode, PyTorch estimator).

Normal-journey demo: no Docker/ECR, just the entry-point scripts in ./src
uploaded via source_dir. Data is pulled straight from s3://stanford-train-data.

Usage:
    export ROLE_ARN=arn:aws:iam::751871643798:role/sagemaker-stanford-exec
    python3 launch_jobs.py            # submit all 4, return immediately
    python3 launch_jobs.py --wait     # block until all 4 finish
    python3 launch_jobs.py --skip goemotions_rl orena_rl
"""
from __future__ import annotations

import argparse
import os
import sys

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ACCOUNT = "751871643798"
REGION = "us-east-1"
BUCKET = "stanford-train-data"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

# SageMaker needs an *execution* role it can assume (trust = sagemaker.amazonaws.com).
# Fill this in with a real role ARN once the role exists (see iam/).
ROLE_ARN = os.environ.get(
    "ROLE_ARN", f"arn:aws:iam::{ACCOUNT}:role/sagemaker-stanford-exec")

# Force UTF-8 in the training container. trl's end-of-run model-card save reads a
# template with Path.read_text() (ascii by default); the Qwen3.5 processor config
# carries non-ASCII bytes that crash it under a POSIX locale. PYTHONUTF8=1 (PEP 540)
# makes the default open()/read_text() encoding UTF-8.
BASE_ENV = {"PYTHONUTF8": "1", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def get_session() -> sagemaker.Session:
    boto = boto3.Session(profile_name="stanford_gpu", region_name=REGION)
    return sagemaker.Session(boto_session=boto, default_bucket=BUCKET)


def make_estimator(entry: str, name: str, instance: str, hps: dict,
                   volume_size: int = 100, environment: dict | None = None) -> PyTorch:
    return PyTorch(
        entry_point=entry,
        source_dir=SRC,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=instance,
        framework_version="2.5.1",
        py_version="py311",
        hyperparameters=hps,
        output_path=f"s3://{BUCKET}/output/",
        base_job_name=name,
        sagemaker_session=get_session(),
        volume_size=volume_size,
        max_run=6 * 3600,
        use_spot_instances=False,
        environment={**BASE_ENV, **(environment or {})},
    )


JOBS = {
    "goemotions_sft": dict(
        entry="goemotions_sft.py", name="goemotions-sft", instance="ml.g4dn.xlarge",
        hps={"epochs": "1.0", "max_train_samples": "5000"},
        inputs={"train": f"s3://{BUCKET}/goemotions/train_sft.jsonl"}),
    "goemotions_rl": dict(
        entry="goemotions_rl.py", name="goemotions-rl", instance="ml.g5.xlarge",
        hps={"max_train_samples": "256"},
        inputs={"train": f"s3://{BUCKET}/goemotions/train_rl.jsonl"}),
    "orena_sft": dict(
        entry="orena_sft.py", name="orena-sft", instance="ml.g5.12xlarge",
        # 9B bf16 SFT does not fit one 24 GB A10G at the reference rank 64/128:
        # step 1 OOM'd ~200 MB over. rank 32/64 + expandable_segments (BASE_ENV)
        # are the memory-forced deltas, recorded in config §8.
        hps={"epochs": "1.0", "max_train_samples": "256", "max_seq_length": "2048",
             "lora_rank": "32", "lora_alpha": "64"},
        volume_size=200,
        # TRL SFTTrainer wraps the VLM in DataParallel on the 4-GPU g5.12xlarge and
        # crashes with a cuda:0-vs-cuda:1 device mismatch in the vision embeddings.
        # Pin to a single GPU for the demo (a real run would use the pytorchddp
        # distribution instead of DataParallel).
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        inputs={"train": f"s3://{BUCKET}/sft/train.jsonl",
                "frames": f"s3://{BUCKET}/curated_frames/"}),
    "orena_rl": dict(
        entry="orena_rl.py", name="orena-rl", instance="ml.g5.2xlarge",
        hps={"max_samples": "256", "max_steps": "50"},
        volume_size=200,
        inputs={"data": f"s3://{BUCKET}/rlvr/frames_train_rl.parquet",
                "frames": f"s3://{BUCKET}/curated_frames/"}),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="block until all jobs finish")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=list(JOBS.keys()), help="jobs to skip")
    args = ap.parse_args()

    launched: dict[str, str] = {}
    for key, cfg in JOBS.items():
        if key in args.skip:
            print(f"SKIP {key}", flush=True)
            continue
        est = make_estimator(cfg["entry"], cfg["name"], cfg["instance"], cfg["hps"],
                             cfg.get("volume_size", 100),
                             environment=cfg.get("environment"))
        est.fit(inputs=cfg["inputs"], wait=args.wait)
        name = est.latest_training_job.name
        launched[key] = name
        print(f"SUBMITTED {key}: {name}  ({cfg['instance']})", flush=True)

    print("\n=== launched training jobs ===", flush=True)
    for k, v in launched.items():
        print(f"{k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
