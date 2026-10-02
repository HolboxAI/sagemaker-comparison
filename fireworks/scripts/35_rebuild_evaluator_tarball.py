#!/usr/bin/env python3
"""Rebuild the GoEmotions evaluator the way `ep upload` does it.

The prior ACTIVE evaluator was created with NO `source` and NO `requirements`
field, and uploaded as a SINGLE .tar.gz tarball (not loose files). Our earlier
two-file upload with source=TYPE_UPLOAD built fine locally but BUILD_FAILED
remotely. This script mirrors the eval_protocol `Evaluator.create()` flow.
"""
import os, time, tarfile, tempfile, shutil, requests
from fireworks import Fireworks

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)
EV = "fs-or1-goemotions-reward"

eval_py = open(f"{BASE}/data/evaluators/ge_evaluation.py").read()
req_txt = open(f"{BASE}/data/evaluators/requirements.txt").read()

# 1. build a clean source dir with just evaluation.py + requirements.txt
tmpdir = tempfile.mkdtemp(prefix="ge_eval_")
srcdir = os.path.join(tmpdir, "ge_eval")
os.makedirs(srcdir, exist_ok=True)
open(os.path.join(srcdir, "evaluation.py"), "w").write(eval_py)
open(os.path.join(srcdir, "requirements.txt"), "w").write(req_txt)

# 2. tar it with the parent dir included (matching eval_protocol's arcname scheme)
tar_name = "ge_eval.tar.gz"
tar_path = os.path.join(tmpdir, tar_name)
with tarfile.open(tar_path, "w:gz") as tar:
    for fn in ("evaluation.py", "requirements.txt"):
        tar.add(os.path.join(srcdir, fn), arcname=f"ge_eval/{fn}")
tar_size = os.path.getsize(tar_path)
print(f"tarball {tar_path} ({tar_size} bytes)")

# 3. delete any prior evaluator
try:
    fw.evaluators.delete(EV)
    print(f"deleted old {EV}")
except Exception as e:
    print(f"delete (ok): {type(e).__name__}")

time.sleep(2)

# 4. create with MINIMAL params (no source, no requirements)
fw.evaluators.create(
    evaluator_id=EV,
    evaluator={
        "display_name": EV,
        "description": "GoEmotions multi-label reward (Jaccard) for Fireworks-vs-SageMaker GRPO",
        "entry_point": "evaluation.py::goemotions_reward",
    },
)
print("create OK (minimal params)")

# 5. single-file upload endpoint for the tarball
ep = fw.evaluators.get_upload_endpoint(EV, filename_to_size={tar_name: str(tar_size)})
urls = getattr(ep, "filename_to_signed_urls", None) or getattr(ep, "filenameToSignedUrls", None)
signed = urls[tar_name]
print("got signed URL for", tar_name)

with open(tar_path, "rb") as f:
    r = requests.put(
        signed,
        data=f,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Goog-Content-Length-Range": f"{tar_size},{tar_size}",
        },
        timeout=600,
    )
print(f"PUT {tar_name} -> {r.status_code}")
r.raise_for_status()

# 6. validate + poll
fw.evaluators.validate_upload(EV, body={})
print("validate OK — building...")
for i in range(30):
    time.sleep(10)
    e = fw.evaluators.get(EV)
    st = getattr(e, "state", None)
    print(f"[{i*10}s] state={st} status={getattr(e,'status',None)}")
    if st in ("ACTIVE", "BUILD_FAILED"):
        break

shutil.rmtree(tmpdir, ignore_errors=True)
