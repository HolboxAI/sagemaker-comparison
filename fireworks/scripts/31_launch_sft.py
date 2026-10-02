#!/usr/bin/env python3
"""Stage 2 — launch the 2 SFT jobs (ORena VLM + GoEmotions text), 1 epoch each.

Logs full job objects to logs/31_sft_launch.log and records each job id to a
run-manifest JSON for the SageMaker parity log.
"""
import os, json, datetime
from fireworks import Fireworks

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

log = open(f"{BASE}/logs/31_sft_launch.log", "w")
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    log.write(line + "\n"); log.flush()

JOBS = [
    dict(
        name="fs-or1-orena-sft",
        dataset=f"accounts/{ACCOUNT}/datasets/fs-or1-orena-sft",
        base_model="accounts/fireworks/models/qwen3p5-9b",
        epochs=1, learning_rate=1e-4, lora_rank=64, max_context_length=32768,
        output_model=f"accounts/{ACCOUNT}/models/fs-or1-orena-sft",
    ),
    dict(
        name="fs-or1-ge-sft",
        dataset=f"accounts/{ACCOUNT}/datasets/fs-or1-ge-sft",
        base_model="accounts/fireworks/models/qwen3-0p6b",
        epochs=1, learning_rate=1e-4, lora_rank=16, max_context_length=8192,
        output_model=f"accounts/{ACCOUNT}/models/fs-or1-ge-sft",
    ),
]

manifest = []
for j in JOBS:
    emit(f"\n===== SFT job {j['name']} =====")
    emit(f"  base={j['base_model']} epochs={j['epochs']} lr={j['learning_rate']} rank={j['lora_rank']} ctx={j['max_context_length']}")
    try:
        resp = fw.supervised_fine_tuning_jobs.create(
            dataset=j["dataset"],
            base_model=j["base_model"],
            epochs=j["epochs"],
            learning_rate=j["learning_rate"],
            lora_rank=j["lora_rank"],
            max_context_length=j["max_context_length"],
            output_model=j["output_model"],
        )
        d = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
        emit(f"  CREATE OK")
        emit(json.dumps(d, default=str, indent=1))
        job_name = d.get("name") or d.get("id") or j["name"]
        manifest.append({**j, "job_name": job_name, "state": d.get("state"), "created": str(datetime.datetime.now(datetime.timezone.utc))})
    except Exception as e:
        emit(f"  CREATE ERR {type(e).__name__}: {str(e)[:800]}")
        manifest.append({**j, "error": f"{type(e).__name__}: {str(e)[:500]}"})

# persist manifest
mf_path = f"{BASE}/metrics/run_manifest.json"
old = {}
if os.path.exists(mf_path):
    try: old = json.load(open(mf_path))
    except Exception: old = {}
old["sft"] = manifest
json.dump(old, open(mf_path, "w"), indent=2)
emit(f"\nmanifest -> {mf_path}")
log.close()
emit("\nDONE (log -> logs/31_sft_launch.log)")
