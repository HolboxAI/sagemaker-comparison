#!/usr/bin/env python3
"""Confirm training access by listing resources with account_id."""
import os, json
from fireworks import Fireworks

ACCOUNT = "purchase-j087ceepz9i"
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

print("account_id:", ACCOUNT)

# Account status (flag provisioning issues)
try:
    accts = fw.accounts.list()
    for a in accts.accounts:
        print(f"[account] name={a.name} state={a.state} status={a.status.code}:{a.status.message[:80]}")
except Exception as e:
    print(f"[account] ERR {type(e).__name__}: {str(e)[:300]}")

for name in ["supervised_fine_tuning_jobs", "dpo_jobs",
             "reinforcement_fine_tuning_jobs", "datasets",
             "evaluators", "evaluation_jobs", "deployments", "lora"]:
    res = getattr(fw, name)
    try:
        lst = res.list()
        print(f"[{name}] list OK —", json.dumps(lst, default=str)[:300])
    except Exception as e:
        print(f"[{name}] list ERR {type(e).__name__}: {str(e)[:300]}")

# Models (VLM + finetuning-eligibility)
print("\n[models] listing with account scope ...")
try:
    ms = fw.models.list()
    data = getattr(ms, "data", ms)
    print(f"[models] total: {len(data)}")
    vlm = [m for m in data if getattr(m, "supports_image_input", False)]
    print(f"[models] VLM-capable: {len(vlm)}")
    for m in vlm:
        print("   -", m.id)
except Exception as e:
    print(f"[models] ERR {type(e).__name__}: {str(e)[:300]}")
