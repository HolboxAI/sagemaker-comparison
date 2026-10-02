#!/usr/bin/env python3
"""Dump the exact pydantic type fields needed to construct all 4 launch calls."""
import os, json
import fireworks.types as T

def dump(cls):
    print(f"\n===== {cls.__name__} =====")
    try:
        fields = cls.model_fields
    except AttributeError:
        print("  (no model_fields)", cls)
        return
    for name, f in fields.items():
        req = "REQUIRED" if f.is_required() else "optional"
        print(f"  {name}: {f.annotation}  [{req}]  default={f.default!r}")

for name in ["DatasetParam", "Dataset", "TrainingConfig", "InferenceParameters",
             "ReinforcementLearningLossConfig", "Evaluator", "EvaluatorCriterion",
             "EvaluatorCriterionCodeSnippets", "WandbConfig"]:
    cls = getattr(T, name, None)
    if cls is None:
        print(f"\n{name}: NOT in fireworks.types")
    else:
        dump(cls)

# list all public type names
print("\n===== all fireworks.types names =====")
print([n for n in dir(T) if not n.startswith("_")])
