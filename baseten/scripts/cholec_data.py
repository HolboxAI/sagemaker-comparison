#!/usr/bin/env python3
"""CholecT50 data pipeline: fetch only the manifest-listed frames from the
remote zip (HTTP range requests via remotezip) and build trl datasets.

The manifests in data/{train,val,test}.jsonl name each frame's zip member path
(e.g. "CholecT50/videos/VID29/001524.png") plus its triplet/component labels.
No full 62.7GB download is needed -- we range-request just the ~3600 frames the
manifests reference (~1.9GB total).
"""
from __future__ import annotations

import json
import os

from datasets import Dataset
from PIL import Image
from remotezip import RemoteZip

from reward import PROMPT

ZIP_URL = "https://s3.unistra.fr/camma_public/datasets/cholect50/CholecT50.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests")


def load_manifest(split: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_frames(members: list[str], cache_dir: str) -> dict[str, str | None]:
    """Download PNG members from the remote zip into cache_dir (idempotent).

    Returns {member: local_path} for every member that resolved; missing members
    are simply absent from the returned dict.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths: dict[str, str] = {}
    missing: list[str] = []
    for m in members:
        p = os.path.join(cache_dir, m)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            paths[m] = p
        else:
            missing.append(m)

    if missing:
        print(f"  fetching {len(missing)} frames from remote zip ...", flush=True)
        with RemoteZip(ZIP_URL) as rz:
            names = set(rz.namelist())
            for m in missing:
                if m not in names:
                    print(f"  WARN: member not in zip: {m}", flush=True)
                    continue
                p = os.path.join(cache_dir, m)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                rz.extract(m, cache_dir)
                paths[m] = p
    return paths


def _load_images(rows: list[dict], cache_dir: str):
    members = [r["member"] for r in rows]
    paths = download_frames(members, cache_dir)
    out = []
    for r in rows:
        p = paths.get(r["member"])
        if not p:
            continue
        out.append((r, Image.open(p).convert("RGB")))
    return out


def build_sft_dataset(split: str, cache_dir: str,
                      max_samples: int | None = None) -> Dataset:
    """trl SFT dataset: plain-string messages + `images` list (auto VLM collator)."""
    rows = load_manifest(split)
    if max_samples:
        rows = rows[:max_samples]
    records = []
    for r, img in _load_images(rows, cache_dir):
        records.append({
            "messages": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": ", ".join(r["triplets"])},
            ],
            "images": [img],
        })
    return Dataset.from_list(records)


def build_grpo_dataset(split: str, cache_dir: str,
                       max_samples: int | None = None) -> Dataset:
    """trl GRPO dataset: conversational `prompt` + `images` list + gold columns.

    The gold columns (triplets/instruments/verbs/targets) are forwarded to the
    reward function by GRPOTrainer, aligned with the generated completions.
    """
    rows = load_manifest(split)
    if max_samples:
        rows = rows[:max_samples]
    records = []
    for r, img in _load_images(rows, cache_dir):
        records.append({
            "prompt": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ]}],
            "images": [img],
            "triplets": r["triplets"],
            "instruments": r["instruments"],
            "verbs": r["verbs"],
            "targets": r["targets"],
        })
    return Dataset.from_list(records)
