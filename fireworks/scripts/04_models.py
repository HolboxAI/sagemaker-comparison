#!/usr/bin/env python3
"""List Fireworks catalog (paginated): name + finetuning eligibility flags."""
import os, json
from fireworks import Fireworks

ACCOUNT = "purchase-j087ceepz9i"
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

all_models = []
page = None
for _ in range(20):
    kw = {"page_token": page} if page else {}
    ms = fw.models.list(**kw)
    models = getattr(ms, "models", None) or getattr(ms, "data", [])
    all_models.extend(models)
    nxt = getattr(ms, "next_page_token", None) or getattr(ms, "nextPageToken", "")
    if not nxt:
        break
    page = nxt

print(f"total models fetched: {len(all_models)}")
print(f"{'NAME':<60} img  lora rl   srvr ctx")
for m in all_models:
    n = (m.name or "").replace("accounts/fireworks/models/", "").replace("accounts/fireworks/routers/", "")
    print(f"{n:<60} {int(m.supports_image_input)}    {int(m.supports_lora)}    {int(m.rl_tunable)}    {int(m.supports_serverless)}    {m.context_length}")

# filters
print("\n=== QWEN models ===")
for m in all_models:
    if "qwen" in (m.name or "").lower():
        print("  ", m.name, "| img", m.supports_image_input, "| lora", m.supports_lora, "| rl", m.rl_tunable)

print("\n=== image-capable ===")
for m in all_models:
    if m.supports_image_input:
        print("  ", m.name, "| lora", m.supports_lora, "| rl", m.rl_tunable)

print("\n=== full dump of all fetched models ===")
for m in all_models:
    d = m.model_dump() if hasattr(m, "model_dump") else str(m)
    # slim it
    slim = {k: d.get(k) for k in ["name","display_name","description","kind","context_length",
            "supports_image_input","supports_lora","rl_tunable","supports_serverless",
            "peft_details","teft_details","base_model_details","state","status","imported_from"]}
    print(json.dumps(slim, default=str, indent=1))
