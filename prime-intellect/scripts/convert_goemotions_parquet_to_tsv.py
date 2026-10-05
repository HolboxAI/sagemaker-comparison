#!/usr/bin/env python3
"""Convert HF go_emotions parquet (simplified config) -> canonical tsv + emotions.txt.

Source: `google-research-datasets/go_emotions` (HF), downloaded with
`hf download google-research-datasets/go_emotions --repo-type dataset`.
The "simplified" parquet holds the canonical splits (train 43,410 / dev 5,426 /
test 5,427) with columns: text (string), labels (list<int64>), id (string).

Output (matches build_goemotions_dataset.py's parser):
    <text> \t <comma-separated label ids, sorted> \t <id>
    emotions.txt  -> 28 names, one per line (id = line index)

Canonical label order (id -> name) is fixed by the original Google release and
matches the one-hot column order of the raw parquet.
"""
import argparse
import os

import pyarrow.parquet as pq

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

SPLIT = {"train": "train", "validation": "dev", "test": "test"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", required=True,
                    help="dir containing simplified/{train,validation,test}-00000-of-00001.parquet")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_label_ids = set()
    for parquet_name, tsv_name in SPLIT.items():
        src = os.path.join(args.in_dir, "simplified",
                           f"{parquet_name}-00000-of-00001.parquet")
        t = pq.read_table(src)
        texts = t["text"].to_pylist()
        labels = t["labels"].to_pylist()
        ids = t["id"].to_pylist()
        dst = os.path.join(args.out_dir, f"{tsv_name}.tsv")
        with open(dst, "w") as f:
            for text, lab, rowid in zip(texts, labels, ids):
                lab_ids = sorted(int(x) for x in lab)
                all_label_ids.update(lab_ids)
                f.write(f"{text}\t{','.join(str(x) for x in lab_ids)}\t{rowid}\n")
        print(f"{tsv_name}.tsv: {t.num_rows} rows")

    with open(os.path.join(args.out_dir, "emotions.txt"), "w") as f:
        f.write("\n".join(EMOTIONS) + "\n")

    print(f"emotions.txt: {len(EMOTIONS)} labels")
    print(f"label ids seen: min={min(all_label_ids)} max={max(all_label_ids)} "
          f"unique={len(all_label_ids)} (expect 0..27)")


if __name__ == "__main__":
    main()
