#!/usr/bin/env python3
"""Build GoEmotions chat (SFT) + rl (GRPO) JSONL for Prime Intellect fine-tuning.

Matches FIREWORKS_4EXP_CONFIG.md §3.2. Input TSV format per line:
    <text> \t <comma-separated label ids> \t <id>
Labels are mapped to canonical names via emotions.txt (id -> name), sorted by id
so the output is deterministic. Multi-label: 0..N labels per example.

Output:
    out/chat/{train,dev}.jsonl   SFT messages with assistant = "joy, gratitude"
    out/rl/{train,dev}.jsonl     GRPO messages + top-level ground_truth.labels
"""
import argparse
import json
import os


def load_emotions(path: str) -> dict:
    name = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) >= 2:
                name[parts[0].strip()] = parts[1].strip()
            else:
                # some versions are just a list of names, one per line
                name[str(len(name))] = parts[0].strip()
    return name


def parse_tsv(path: str, emotions: dict, max_rows: int) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text, label_ids, _ = line.split("\t", 2)
            labels = []
            for lid in label_ids.split(","):
                lid = lid.strip()
                name = emotions.get(lid, lid)
                if name not in labels:
                    labels.append(name)
            labels.sort()
            rows.append((text, labels))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


SYSTEM = ("Classify the text into one or more of the 28 GoEmotions labels. "
          "Reply with a comma-separated list of label names only.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--dev_tsv", required=True)
    ap.add_argument("--test_tsv", default=None, help="optional; unused by these 4 experiments")
    ap.add_argument("--emotions", required=True)
    ap.add_argument("--out", default="./goemotions")
    ap.add_argument("--max_train_rows", type=int, default=0)
    ap.add_argument("--max_dev_rows", type=int, default=0)
    args = ap.parse_args()

    emo = load_emotions(args.emotions)
    for mode in ("chat", "rl"):
        os.makedirs(os.path.join(args.out, mode), exist_ok=True)
    for split, src, cap in (("train", args.train_tsv, args.max_train_rows),
                            ("dev", args.dev_tsv, args.max_dev_rows)):
        rows = parse_tsv(src, emo, cap)
        chat_path = os.path.join(args.out, "chat", f"{split}.jsonl")
        rl_path = os.path.join(args.out, "rl", f"{split}.jsonl")
        with open(chat_path, "w") as fc, open(rl_path, "w") as fr:
            for text, labels in rows:
                chat = {"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": ", ".join(labels)},
                ]}
                rl = {"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
                    "ground_truth": {"labels": labels},
                    "input_metadata": {}}
                fc.write(json.dumps(chat, ensure_ascii=False) + "\n")
                fr.write(json.dumps(rl, ensure_ascii=False) + "\n")
        print(f"{split}: wrote {len(rows)} rows -> {chat_path} and {rl_path}")


if __name__ == "__main__":
    main()
