#!/usr/bin/env python3
"""Isolate the GoEmotions evaluator BUILD_FAILED.

Builds a MINIMAL evaluator (trivial reward, no input_dataset, no completion_params)
to decide whether the failure is in my ge_evaluation.py code or in the remote
eval-protocol build environment.
"""
import os, time, httpx, tempfile
from fireworks import Fireworks
from fireworks.types.evaluator_create_params import Evaluator, EvaluatorSourceParam

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)
EV = "fs-or1-minimal-test"

MINIMAL = '''from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test


@evaluation_test(
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def minimal_reward(row: EvaluationRow, **kwargs) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(score=1.0, is_score_valid=True, reason="minimal")
    return row
'''
REQ = "eval-protocol\n"

# delete any prior
try:
    fw.evaluators.delete(EV)
    print(f"deleted old {EV}")
except Exception as e:
    print(f"delete (ok): {type(e).__name__}")

time.sleep(2)

fw.evaluators.create(
    evaluator=Evaluator(
        display_name=EV,
        description="minimal evaluator to isolate build failure",
        entry_point="evaluation.py::minimal_reward",
        requirements="",
        source=EvaluatorSourceParam(type="TYPE_UPLOAD"),
        criteria=[],
    ),
    evaluator_id=EV,
)
print("create OK")

files = {"evaluation.py": MINIMAL.encode(), "requirements.txt": REQ.encode()}
size_map = {fn: str(len(b)) for fn, b in files.items()}
ep = fw.evaluators.get_upload_endpoint(EV, filename_to_size=size_map)
urls = getattr(ep, "filename_to_signed_urls", None) or getattr(ep, "filenameToSignedUrls", None)
for fn, body in files.items():
    n = len(body)
    r = httpx.put(urls[fn], content=body, headers={"Content-Type": "application/octet-stream", "X-Goog-Content-Length-Range": f"{n},{n}"}, timeout=60)
    print(f"PUT {fn} -> {r.status_code}")
    r.raise_for_status()

fw.evaluators.validate_upload(EV, body={})
print("validate OK — building...")

for i in range(24):
    time.sleep(10)
    e = fw.evaluators.get(EV)
    st = getattr(e, "state", None)
    print(f"[{i*10}s] state={st}")
    if st in ("ACTIVE", "BUILD_FAILED"):
        break
