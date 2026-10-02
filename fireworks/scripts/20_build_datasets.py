#!/usr/bin/env python3
"""Build the 4 JSONL datasets for the Fireworks/SageMaker 4-experiment run.

Outputs (all under data/):
  goemotions/goemotions_sft.jsonl      CHAT  (full train, 43,410)
  goemotions/goemotions_rl.jsonl       RL    (5,000 subset)
  orena/orena_sft.jsonl                CHAT  (full train, 10,980)
  orena/orena_rl.jsonl                 RL    (2,000 subset)
Also prints answer-format samples so the reward evaluators can normalize correctly.
"""
import json, os
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/data/goemotions", exist_ok=True)
os.makedirs(f"{BASE}/data/orena", exist_ok=True)

# ---- GoEmotions ----
emotions = open(f"{BASE}/data/goemotions/data/emotions.txt").read().splitlines()
assert len(emotions) == 28, f"expected 28 labels, got {len(emotions)}"
LABEL_NAMES = ", ".join(emotions)

GE_SYS = ("Classify the text into one or more of the 28 GoEmotions emotion labels. "
          "Reply with a comma-separated list of label names only, and nothing else. "
          f"Valid labels: {LABEL_NAMES}.")

def parse_ge(line):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3:
        text = "\t".join(parts[:-2])
        labels = parts[-2]
    else:
        text, labels = parts[0], parts[1]
    ids = [int(x) for x in labels.split(",")]
    names = [emotions[i] for i in ids]
    return text, names

ge_sft_n = 0
ge_rl_n = 0
GE_RL_LIMIT = 5000
with open(f"{BASE}/data/goemotions/goemotions_sft.jsonl", "w") as fs, \
     open(f"{BASE}/data/goemotions/goemotions_rl.jsonl", "w") as fr:
    for line in open(f"{BASE}/data/goemotions/data/train.tsv"):
        if not line.strip():
            continue
        text, names = parse_ge(line)
        answer = ", ".join(names)
        fs.write(json.dumps({"messages": [
            {"role": "system", "content": GE_SYS},
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]}) + "\n")
        ge_sft_n += 1
        if ge_rl_n < GE_RL_LIMIT:
            fr.write(json.dumps({"messages": [
                {"role": "system", "content": GE_SYS},
                {"role": "user", "content": text},
            ], "ground_truth": {"labels": names}, "input_metadata": {}}) + "\n")
            ge_rl_n += 1

# ---- ORena ----
ORENA_SRC = f"{BASE}/reference/evaluator_orena/orena-focus-rft/data/orena_train_full.jsonl"
ORENA_RL_LIMIT = 2000
orena_sft_n = 0
orena_rl_n = 0
fmt_counter = Counter()
fmt_samples = defaultdict(list)
with open(f"{BASE}/data/orena/orena_sft.jsonl", "w") as fs, \
     open(f"{BASE}/data/orena/orena_rl.jsonl", "w") as fr:
    for line in open(ORENA_SRC):
        if not line.strip():
            continue
        d = json.loads(line)
        gt = d["ground_truth"]
        fmt = gt["answer_format"]
        ans = gt["answer"]
        fmt_counter[fmt] += 1
        if len(fmt_samples[fmt]) < 5:
            fmt_samples[fmt].append(ans)

        # SFT: append assistant answer to the existing [system, user] messages
        sft = {"messages": d["messages"] + [{"role": "assistant", "content": ans}]}
        fs.write(json.dumps(sft) + "\n")
        orena_sft_n += 1

        # RL: keep as-is, subset
        if orena_rl_n < ORENA_RL_LIMIT:
            fr.write(line if line.endswith("\n") else line + "\n")
            orena_rl_n += 1

print("=== built ===")
print(f"goemotions_sft.jsonl  {ge_sft_n} rows (CHAT)")
print(f"goemotions_rl.jsonl   {ge_rl_n} rows (RL)")
print(f"orena_sft.jsonl       {orena_sft_n} rows (CHAT)")
print(f"orena_rl.jsonl        {orena_rl_n} rows (RL)")
print("\n=== ORena answer_format distribution (full train) ===")
for k, v in fmt_counter.most_common():
    print(f"  {k}: {v}")
print("\n=== ORena answer samples by format ===")
for k, vs in fmt_samples.items():
    print(f"  [{k}]")
    for v in vs:
        print(f"      {v!r}")
print("\n=== GoEmotions label sample ===")
for i in range(0, len(emotions), 7):
    print("  ", list(enumerate(emotions))[i:i+7])
