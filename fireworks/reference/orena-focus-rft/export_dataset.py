"""Materialize the ORena-FOCUS RL training distribution as Fireworks RFT JSONL.

This is the piece that has no counterpart on Prime. The verifiers environment
built prompts lazily *inside* the rollout worker, so it could stream a private
HF dataset and downscale frames at rollout time. Fireworks RFT takes a static
JSONL dataset instead: every frame has to be fetched, downscaled and base64
encoded here, offline, and shipped inside the file.

Consequences worth knowing before you run this:

  * `image_max_side` is baked in. Changing it means re-exporting and re-uploading,
    not editing a config. Pick it once (see the size table in the README).
  * The private HF dataset is read HERE, on your machine, with your local
    `huggingface-cli login`. No HF_TOKEN needs to exist on Fireworks — which
    removes the failure mode that killed step 0 on Prime.
  * Filters are applied here too, so the JSONL *is* the training distribution.

Same filters, same system prompt, same scorer as `orena-focus-rl-v2`. If you
change a filter, change it in both places or the two platforms stop being
comparable.

    uv run export_dataset.py --split train      --max-rows 2000 --out data/orena_train.jsonl
    uv run export_dataset.py --split validation --max-rows 200  --out data/orena_eval.jsonl

Add `--no-filters` to export the unfiltered base distribution — that is the eval
set you report from, exactly as on Prime.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from collections import Counter

import orena_scoring as S

# `datasets` and `pillow` are imported lazily, inside the functions that use
# them. `ep upload` tars this whole directory and ships it to the evaluator
# build, which installs eval_protocol but not the export-time dependencies —
# a module-level import here would break collection of a file that the remote
# never needs to run.

# Kept byte-identical to orena_focus_rl_v2.py. See that file for why each format
# and class is dropped; duplicating the rationale here would let the two drift.
TRAIN_FORMATS = ["number", "FO_Class", "binary"]
EXCLUDE_CLASSES = ["silicone loop"]

SYSTEM_PROMPT = """You are a surgical video analyst. You are shown a single frame from a \
laparoscopic or endoscopic procedure and asked one question about the surgical \
FOREIGN OBJECTS visible in it.

The only foreign object classes that exist are:
Clip, Sponge, Silicone loop, External drain, Specimen, Specimen bag, Needle, Gallstone.
Never invent a class outside this list. Surgical instruments (graspers, scissors, \
trocars, cautery hooks) are NOT foreign objects and must never be counted or named.

Work through the frame quadrant by quadrant, noting every candidate object and its \
position, discarding instruments and anatomy. Keep your reasoning brief.

Your final reply must be ONLY the answer itself — no explanation, no units, no \
restating of the question, no extra words.

Answer formats:
- "how many ..."          -> a single integer, e.g. 3
- "yes or no"             -> exactly one word: yes or no
- "which class(es) ..."   -> comma-separated class names, e.g. Clip, Sponge  (or: none)
"""


def to_data_url(img, max_side: int, quality: int) -> str:
    """PIL image (or the raw dict the HF Image feature yields) -> base64 JPEG data URL.

    Fireworks accepts PNG and JPEG only, and rejects plain http(s) URLs in
    training data, so the bytes must travel inline.
    """
    from PIL import Image

    if max_side and max_side > 0:
        if isinstance(img, dict) and img.get("bytes"):
            img = Image.open(io.BytesIO(img["bytes"]))
        im = img.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            sc = max_side / max(w, h)
            im = im.resize((int(w * sc), int(h * sc)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
    elif isinstance(img, dict) and img.get("bytes"):
        raw = img["bytes"]
    else:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def canonical_classes(names) -> set[str]:
    import re

    out: set[str] = set()
    for n in names or []:
        c = re.sub(r"\s+", " ", str(n).strip().lower())
        out.add(S.CLASS_ALIASES.get(c, c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default="om-chikhaliya/orena-focus-frame-vqa")
    ap.add_argument("--local-dir", default=os.environ.get("ORENA_DATASET_DIR"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=-1, help="-1 exports everything that survives filtering")
    ap.add_argument("--image-max-side", type=int, default=512)
    ap.add_argument("--image-quality", type=int, default=85)
    ap.add_argument(
        "--no-filters",
        action="store_true",
        help="Export the unfiltered base distribution — all 5 answer formats, no class exclusions. "
        "Use this for the honest eval set.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed; 0 keeps dataset order.")
    args = ap.parse_args()

    from datasets import load_dataset, load_from_disk

    formats = [] if args.no_filters else TRAIN_FORMATS
    excluded = set() if args.no_filters else canonical_classes(EXCLUDE_CLASSES)

    if args.local_dir:
        d = load_from_disk(args.local_dir)
        ds = d[args.split] if hasattr(d, "keys") else d
    else:
        ds = load_dataset(args.dataset_id, split=args.split)

    n_start = len(ds)
    if formats:
        wanted = {S.canonical_format(f) for f in formats}
        ds = ds.filter(lambda r: S.canonical_format(r["answer_format"]) in wanted)
    n_fmt = len(ds)

    if excluded:
        # Drops rows that *supervise* an excluded class — checked against the
        # question too, since a `number` row asking "how many silicone loops?"
        # has an integer gold that never names the class. Removes supervision,
        # not pixels: those frames still appear under other questions.
        ds = ds.filter(
            lambda r: not ((S.parse_class_set(r["answer"]) | S.parse_class_set(r["question"])) & excluded)
        )
    n_cls = len(ds)

    if args.seed:
        ds = ds.shuffle(seed=args.seed)

    # Slice BEFORE encoding. Base64-encoding a frame is by far the slowest step
    # here; doing it for rows that will never be written is pure waste.
    if args.max_rows and args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    print(
        f"[export] {args.split}: {n_start} -> {n_fmt} after format filter "
        f"{sorted(formats) if formats else '(none)'} -> {n_cls} after excluding "
        f"{sorted(excluded) if excluded else '(nothing)'} -> {len(ds)} after cap "
        f"({args.max_rows if args.max_rows > 0 else 'uncapped'}).",
        file=sys.stderr,
    )
    if len(ds) == 0:
        print("[export] FATAL: every row was filtered out.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fmt_counts: Counter[str] = Counter()
    total_bytes = 0

    with open(args.out, "w") as f:
        for i, row in enumerate(ds):
            url = to_data_url(row["image"], args.image_max_side, args.image_quality)
            fmt = S.canonical_format(row["answer_format"])
            fmt_counts[fmt] += 1
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": row["question"]},
                        ],
                    },
                ],
                # Everything the scorer needs travels with the row. The evaluator
                # is stateless: it never reaches back to HF, so a token expiring
                # cannot silently zero the reward mid-run.
                "ground_truth": {
                    "answer": row["answer"],
                    "answer_format": fmt,
                    "choices": list(row.get("choices") or []),
                },
                "input_metadata": {
                    "row_id": f"{row['video_id']}:{row['frame_id']}:{i}",
                    "dataset_info": {
                        "video_id": row["video_id"],
                        "frame_id": row["frame_id"],
                        "question": row["question"],
                        "answer_format": fmt,
                    },
                },
            }
            line = json.dumps(rec)
            total_bytes += len(line) + 1
            f.write(line + "\n")
            if (i + 1) % 250 == 0:
                print(f"[export]   {i + 1}/{len(ds)} rows, {total_bytes / 1e6:.0f} MB", file=sys.stderr)

    print(
        f"[export] wrote {len(ds)} rows to {args.out} "
        f"({total_bytes / 1e6:.1f} MB, {total_bytes / len(ds) / 1e3:.0f} KB/row) "
        f"format mix: {dict(fmt_counts)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
