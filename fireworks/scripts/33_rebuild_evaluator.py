#!/usr/bin/env python3
"""Rebuild the GoEmotions evaluator after BUILD_FAILED (add mcp<2 pin)."""
import os, time, httpx
from fireworks import Fireworks
from fireworks.types.evaluator_create_params import Evaluator, EvaluatorSourceParam

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)
EV = "fs-or1-goemotions-reward"

eval_py = open(f"{BASE}/data/evaluators/ge_evaluation.py").read()
req_txt = open(f"{BASE}/data/evaluators/requirements.txt").read()

# delete any prior failed evaluator
try:
    fw.evaluators.delete(EV)
    print(f"deleted old {EV}")
except Exception as e:
    print(f"delete (ok if not found): {type(e).__name__} {str(e)[:120]}")

time.sleep(2)

# recreate — requirements field left empty to match the prior working evaluator;
# the requirements.txt file in the source drives the container install.
try:
    fw.evaluators.create(
        evaluator=Evaluator(
            display_name=EV,
            description="GoEmotions multi-label reward (Jaccard) for Fireworks-vs-SageMaker GRPO",
            entry_point="evaluation.py::goemotions_reward",
            requirements="",
            source=EvaluatorSourceParam(type="TYPE_UPLOAD"),
            criteria=[],
        ),
        evaluator_id=EV,
    )
    print(f"recreate OK")
except Exception as e:
    print(f"recreate: {type(e).__name__} {str(e)[:300]}")

files = {"evaluation.py": eval_py.encode(), "requirements.txt": req_txt.encode()}
size_map = {fn: str(len(b)) for fn, b in files.items()}
ep = fw.evaluators.get_upload_endpoint(EV, filename_to_size=size_map)
urls = getattr(ep, "filename_to_signed_urls", None) or getattr(ep, "filenameToSignedUrls", None)
for fn, body in files.items():
    n = len(body)
    r = httpx.put(urls[fn], content=body, headers={"Content-Type": "application/octet-stream", "X-Goog-Content-Length-Range": f"{n},{n}"}, timeout=60)
    print(f"PUT {fn} -> {r.status_code}")
    r.raise_for_status()

fw.evaluators.validate_upload(EV, body={})
print("validate_upload OK — building...")

# poll build state up to ~3 min
for i in range(18):
    time.sleep(10)
    e = fw.evaluators.get(EV)
    st = getattr(e, "state", None)
    print(f"  [{i*10}s] state={st} status={getattr(e,'status',None)}")
    if st in ("ACTIVE", "BUILD_FAILED"):
        break
