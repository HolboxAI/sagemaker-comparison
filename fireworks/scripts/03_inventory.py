#!/usr/bin/env python3
"""Full dump of existing Fireworks resources on this account."""
import os, json
from fireworks import Fireworks

ACCOUNT = "purchase-j087ceepz9i"
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

def dump_items(res_name, plural):
    res = getattr(fw, res_name)
    try:
        lst = res.list()
        items = getattr(lst, plural, None)
        if items is None:
            items = getattr(lst, "data", []) or []
        print(f"\n===== {res_name} ({len(items)}) =====")
        for it in items:
            d = None
            if hasattr(it, "model_dump"):
                d = it.model_dump()
            elif hasattr(it, "dict"):
                d = it.dict()
            else:
                d = {"repr": str(it)}
            print(json.dumps(d, default=str, indent=1)[:3000])
    except Exception as e:
        print(f"\n===== {res_name} ===== ERR {type(e).__name__}: {str(e)[:300]}")

dump_items("supervised_fine_tuning_jobs", "supervised_fine_tuning_jobs")
dump_items("dpo_jobs", "dpo_jobs")
dump_items("reinforcement_fine_tuning_jobs", "reinforcement_fine_tuning_jobs")
dump_items("datasets", "datasets")
dump_items("evaluators", "evaluators")
dump_items("evaluation_jobs", "evaluation_jobs")
dump_items("deployments", "deployments")
dump_items("lora", "deployed_models")
