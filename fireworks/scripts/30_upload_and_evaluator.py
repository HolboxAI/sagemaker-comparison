#!/usr/bin/env python3
"""Stage 1 — upload the 4 datasets + create the GoEmotions reward evaluator.

No training cost. Safe to re-run (create calls are idempotent-ish; we print any
conflict and continue). Logs everything to logs/30_stage1.log.
"""
import os, sys, json, time
from fireworks import Fireworks
from fireworks.types import DatasetParam
from fireworks.types.evaluator_create_params import Evaluator, EvaluatorSourceParam

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

log = open(f"{BASE}/logs/30_stage1.log", "w")
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    log.write(line + "\n"); log.flush()

def full_ds(name): return f"accounts/{ACCOUNT}/datasets/{name}"
def full_ev(name): return f"accounts/{ACCOUNT}/evaluators/{name}"

DATASETS = [
    # (short_id, format, path, example_count)
    ("fs-or1-orena-sft", "CHAT", f"{BASE}/data/orena/orena_sft.jsonl", 10980),
    ("fs-or1-orena-rl",  "RL",   f"{BASE}/data/orena/orena_rl.jsonl",  2000),
    ("fs-or1-ge-sft",    "CHAT", f"{BASE}/data/goemotions/goemotions_sft.jsonl", 43410),
    ("fs-or1-ge-rl",     "RL",   f"{BASE}/data/goemotions/goemotions_rl.jsonl",  5000),
]

# ---- 1. datasets ----
for short, fmt, path, n in DATASETS:
    emit(f"\n===== dataset {short} ({fmt}, {n} rows) =====")
    try:
        fw.datasets.create(dataset=DatasetParam(format=fmt, example_count=str(n), display_name=short),
                           dataset_id=short)
        emit(f"  create OK -> {full_ds(short)}")
    except Exception as e:
        emit(f"  create: {type(e).__name__}: {str(e)[:300]}")
    try:
        with open(path, "rb") as f:
            up = fw.datasets.upload(short, file=f)
        emit(f"  upload OK -> {getattr(up,'state',None) or getattr(up,'status',None)}")
    except Exception as e:
        emit(f"  upload: {type(e).__name__}: {str(e)[:300]}")

# ---- 2. GoEmotions evaluator ----
emit("\n===== evaluator fs-or1-goemotions-reward =====")
EV_ID = "fs-or1-goemotions-reward"
eval_py = open(f"{BASE}/data/evaluators/ge_evaluation.py").read()
req_txt = open(f"{BASE}/data/evaluators/requirements.txt").read()
try:
    fw.evaluators.create(
        evaluator=Evaluator(
            display_name="fs-or1-goemotions-reward",
            description="GoEmotions multi-label reward (Jaccard) for the Fireworks-vs-SageMaker GRPO run",
            entry_point="evaluation.py::goemotions_reward",
            requirements="eval-protocol",
            source=EvaluatorSourceParam(type="TYPE_UPLOAD"),
            criteria=[],
        ),
        evaluator_id=EV_ID,
    )
    emit(f"  create OK -> {full_ev(EV_ID)}")
except Exception as e:
    emit(f"  create: {type(e).__name__}: {str(e)[:300]}")

try:
    size_map = {"evaluation.py": str(len(eval_py)), "requirements.txt": str(len(req_txt))}
    ep = fw.evaluators.get_upload_endpoint(EV_ID, filename_to_size=size_map)
    urls = getattr(ep, "filename_to_signed_urls", None) or getattr(ep, "filenameToSignedUrls", None)
    emit(f"  upload endpoint -> {list(urls.keys()) if urls else ep}")
    if urls:
        import httpx
        for fn, url in urls.items():
            body = eval_py.encode() if fn == "evaluation.py" else req_txt.encode()
            r = httpx.put(url, content=body, headers={"Content-Type": "application/octet-stream"})
            emit(f"  PUT {fn} -> HTTP {r.status_code}")
except Exception as e:
    emit(f"  upload endpoint/PUT: {type(e).__name__}: {str(e)[:300]}")

try:
    v = fw.evaluators.validate_upload(EV_ID, body={})
    emit(f"  validate_upload OK -> {v}")
except Exception as e:
    emit(f"  validate_upload: {type(e).__name__}: {str(e)[:300]}")

emit("\n===== current state =====")
for short, fmt, path, n in DATASETS:
    try:
        d = fw.datasets.get(short)
        emit(f"  dataset {short}: state={getattr(d,'state',None)} format={getattr(d,'format',None)} count={getattr(d,'example_count',None)}")
    except Exception as e:
        emit(f"  dataset {short}: ERR {type(e).__name__} {str(e)[:120]}")
try:
    e = fw.evaluators.get(EV_ID)
    emit(f"  evaluator {EV_ID}: state={getattr(e,'state',None)} status={getattr(e,'status',None)}")
except Exception as ex:
    emit(f"  evaluator {EV_ID}: ERR {type(ex).__name__} {str(ex)[:120]}")

log.close()
emit("\nDONE (log -> logs/30_stage1.log)")
