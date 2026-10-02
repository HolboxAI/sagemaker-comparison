#!/usr/bin/env python3
"""Discover account_id and the method surface of each training resource."""
import os, json
from fireworks import Fireworks  # new import path (avoids deprecation)

fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])

def methods(res):
    return [a for a in dir(res) if not a.startswith("_")]

print("[accounts] methods:", methods(fw.accounts))
print("[users] methods:", methods(fw.users))

# Try to find account id
print("\n[accounts.list] ...")
try:
    lst = fw.accounts.list()
    print("  OK:", json.dumps(lst, default=str)[:1000])
except Exception as e:
    print(f"  ERR {type(e).__name__}: {str(e)[:400]}")

print("\n[accounts.retrieve_current] ...")
for m in ["retrieve_current", "get", "me", "current", "retrieve"]:
    if hasattr(fw.accounts, m):
        try:
            r = getattr(fw.accounts, m)()
            print(f"  {m} OK:", json.dumps(r, default=str)[:1000])
        except Exception as e:
            print(f"  {m} ERR {type(e).__name__}: {str(e)[:300]}")

# Method surface of training resources
for name in ["supervised_fine_tuning_jobs", "dpo_jobs",
             "reinforcement_fine_tuning_jobs", "reinforcement_fine_tuning_steps",
             "datasets", "evaluators", "evaluation_jobs", "deployments", "lora"]:
    res = getattr(fw, name, None)
    print(f"\n[{name}] methods:", methods(res) if res else "MISSING")
