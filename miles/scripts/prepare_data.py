#!/usr/bin/env python
"""Build Miles-format RL prompt data from the canonical `data/` files.

Miles loads prompts from a JSONL via `--prompt-data <file> --input-key <k>
--label-key <k> --apply-chat-template`. A line is:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}],
     "label": "<json string ground truth>"}

`messages` is the full chat message list (system + user). `label` is the ground
truth as a JSON string, because Miles `Sample.label` is a `str` (one key only).
The reward functions parse that string back into a dict / list.

Outputs:
    miles/data/goemotions_rl_miles.jsonl   (from committed data/, fully runnable)
    miles/data/orena_rl_miles_sample.jsonl (2 rows, smoke only)

The full ORena RL file (`data/orena/orena_rl.jsonl`, 2000 rows, 66 MB of base64
images) is not committed. Regenerate it first with the canonical script, then
pass it here with `--orena-input`.
"""
from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOEMOTIONS_SYSTEM = ("Classify the text into one or more of the 28 GoEmotions "
                     "labels. Reply with a comma-separated list of label names only.")

ORENA_SYSTEM = ("You are a surgical VQA assistant. Answer the question using the "
                "image. Keep the answer short and in the requested format.")


def goemotions_rows(src: str) -> list[dict]:
    """canonical goemotions_rl.jsonl -> miles lines (system + user, label list)."""
    out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["messages"]
            # Keep only system + final user turn (the prompt). No assistant turn.
            prompt = [m for m in messages if m["role"] in ("system", "user")]
            labels = row["ground_truth"]["labels"]
            out.append({"messages": prompt, "label": json.dumps(labels)})
    return out


def orena_rows(src: str) -> list[dict]:
    """canonical orena_rl.jsonl -> miles lines.

    The image stays as a base64 data URL inside the user content, matching the
    Fireworks/SageMaker prompt. The VLM generate function (generate_orena.py)
    passes it to SGLang. `label` is the full ground-truth dict as a JSON string.
    """
    out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = [m for m in row["messages"] if m["role"] in ("system", "user")]
            gt = row["ground_truth"]
            out.append({"messages": prompt, "label": json.dumps(gt)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(REPO, "miles", "data"))
    ap.add_argument("--goemotions-input",
                    default=os.path.join(REPO, "data", "goemotions", "goemotions_rl.jsonl"))
    ap.add_argument("--orena-input",
                    default=os.path.join(REPO, "data", "orena", "orena_rl_sample_2rows.jsonl"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ge_out = os.path.join(args.out_dir, "goemotions_rl_miles.jsonl")
    rows = goemotions_rows(args.goemotions_input)
    with open(ge_out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"goemotions: {len(rows)} rows -> {ge_out}")

    orena_out = os.path.join(args.out_dir, "orena_rl_miles_sample.jsonl")
    orows = orena_rows(args.orena_input)
    with open(orena_out, "w", encoding="utf-8") as f:
        for r in orows:
            f.write(json.dumps(r) + "\n")
    print(f"orena: {len(orows)} rows -> {orena_out} (sample only; regenerate full data first)")


if __name__ == "__main__":
    main()
