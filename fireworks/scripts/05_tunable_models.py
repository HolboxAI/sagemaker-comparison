#!/usr/bin/env python3
"""Find the tunable base models (SFT + RL eligible) and their text/VLM split."""
import os, json
from fireworks import Fireworks

ACCOUNT = "purchase-j087ceepz9i"
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

print("[models resource methods]:", [a for a in dir(fw.models) if not a.startswith("_")])

# Try to fetch the known VLM base model directly
for mid in ["accounts/fireworks/models/qwen3-vl-8b-instruct",
            "accounts/fireworks/models/qwen3-vl-8b"]:
    try:
        m = fw.models.get(name=mid)
        d = m.model_dump() if hasattr(m, "model_dump") else str(m)
        slim = {k: d.get(k) for k in ["name","display_name","supports_image_input",
                "supports_lora","rl_tunable","supports_serverless","context_length","kind","state"]}
        print(f"\n[get {mid}]:", json.dumps(slim, default=str))
    except Exception as e:
        print(f"\n[get {mid}] ERR {type(e).__name__}: {str(e)[:200]}")

# Try list() without account scope (public tunable catalog?)
print("\n[list without account_id] ...")
try:
    fw2 = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
    ms = fw2.models.list()
    models = getattr(ms, "models", None) or getattr(ms, "data", [])
    print(f"  total: {len(models)}")
    for m in models[:80]:
        n = (m.name or "").replace("accounts/fireworks/models/","")
        print(f"  {n:<45} img={int(m.supports_image_input)} lora={int(m.supports_lora)} rl={int(m.rl_tunable)} ctx={m.context_length}")
except Exception as e:
    print(f"  ERR {type(e).__name__}: {str(e)[:300]}")
