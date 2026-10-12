#!/usr/bin/env python3
"""Together AI feasibility probe for the CholecT50 SFT -> RLVR project.

Checks, against the public Together API:
  1. the key is valid and fine-tuning is reachable
  2. whether any Qwen3-VL / Qwen2.5-VL / Qwen2-VL models exist (inference catalog)
  3. the current fine-tune job list (empty = nothing has run)

What the API does NOT expose — and we had to read from the docs — is the set of
fine-tuning methods (SFT + DPO only) and the fact that RL/GRPO lives in a separate
"Custom Training" beta. See config/TOGETHER_FINDINGS.md for the conclusions.

Usage:
    export TOGETHER_API_KEY=...
    python probe_together.py
"""
import json
import os
import urllib.request

API = "https://api.together.xyz/v1"
KEY = os.environ.get("TOGETHER_API_KEY", "")


def get(path: str) -> dict:
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    if not KEY:
        raise SystemExit("set TOGETHER_API_KEY first")

    models = get("/models")
    ids = [m["id"] for m in models]
    print(f"models listed: {len(ids)}")

    # The model we want (Qwen3-VL-4B-Instruct) and its family.
    for pat in ("Qwen3-VL", "Qwen2.5-VL", "Qwen2-VL"):
        hits = [i for i in ids if pat.lower() in i.lower()]
        print(f"  {pat}: {hits or 'NONE'}")

    ft = get("/fine-tunes")
    print(f"fine-tunes (existing jobs): {len(ft.get('data', []))}")


if __name__ == "__main__":
    main()
