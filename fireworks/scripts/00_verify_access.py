#!/usr/bin/env python3
"""Verify Fireworks key: inference + training access, and discover SDK surface.

Logs everything to stdout; caller tees to a log file.
"""
import os
import json
import fireworks
from fireworks.client import Fireworks

print("fireworks-ai version:", fireworks.__version__)

api_key = os.environ.get("FIREWORKS_API_KEY")
if not api_key:
    raise SystemExit("FIREWORKS_API_KEY not set")

fw = Fireworks(api_key=api_key)

# --- SDK surface -----------------------------------------------------------
print("\n[client attrs]", [a for a in dir(fw) if not a.startswith("_")])

# --- inference check -------------------------------------------------------
print("\n[inference] listing models ...")
try:
    models = fw.models.list()
    ids = [m.id for m in models.data]
    print(f"[inference] OK — {len(ids)} models. first 5: {ids[:5]}")
    # count VLM-capable models
    vlms = [m.id for m in models.data if getattr(m, "supports_image_input", False)]
    print(f"[inference] VLM-capable models: {len(vlms)}")
    for v in vlms[:15]:
        print("   -", v)
except Exception as e:
    print(f"[inference] ERR {type(e).__name__}: {str(e)[:400]}")

# --- fine-tuning (managed SFT/DPO) -----------------------------------------
print("\n[fine-tuning] discovering surface ...")
ft = getattr(fw, "fine_tuning", None)
if ft is None:
    print("[fine-tuning] no 'fine_tuning' attribute on client")
else:
    print("[fine-tuning] attrs:", [a for a in dir(ft) if not a.startswith("_")])
    jobs = getattr(ft, "jobs", None)
    if jobs is not None:
        print("[fine-tuning] jobs attrs:", [a for a in dir(jobs) if not a.startswith("_")])
        try:
            lst = jobs.list()
            n = getattr(lst, "data", lst)
            print(f"[fine-tuning] list jobs OK — {len(n) if hasattr(n, '__len__') else '?'} jobs")
        except Exception as e:
            print(f"[fine-tuning] list jobs ERR {type(e).__name__}: {str(e)[:500]}")

# --- account / billing -----------------------------------------------------
print("\n[account] ...")
try:
    acct = fw.accounts.retrieve()
    print("[account] OK:", json.dumps(acct, default=str)[:800])
except Exception as e:
    print(f"[account] ERR {type(e).__name__}: {str(e)[:500]}")

# --- training API (RL) module if present -----------------------------------
print("\n[training-api] checking fireworks.training module ...")
try:
    import fireworks.training as tr
    print("[training-api] module present; attrs:", [a for a in dir(tr) if not a.startswith("_")][:40])
except Exception as e:
    print(f"[training-api] module ERR {type(e).__name__}: {str(e)[:300]}")

print("\nDONE")
