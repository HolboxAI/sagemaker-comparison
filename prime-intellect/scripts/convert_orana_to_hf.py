#!/usr/bin/env python3
"""Convert the ORena parquets (frames_{train,test}_true.parquet) into the JSONL
formats Prime Intellect fine-tuning consumes, matching FIREWORKS_4EXP_CONFIG.md §3.1.

Two output modes (--mode):
  chat  -> SFT: system + user(image + question) + assistant(reference answer)
  rl    -> GRPO: system + user(image + question) + top-level ground_truth

Image encoding: base64-inline data-URL inside the user turn (the Fireworks §3.1
form, self-contained single file). The image-content key is Qwen-native
({"type":"image","image": …}); if prime-rl's renderer wants OpenAI-style
({"type":"image_url","image_url":{"url": …}}), swap with a one-line re-map at
run time (see 02_DATA_ORENA.md §1).

Frame resolution replicates the parent repo's train_qwen3vl.load_rows exactly
(video + timestamp_start -> frame_HHMMSS.jpg, nearest-frame fallback within
--max_frame_drift seconds, source-filtered), so it stays byte-identical to our
existing training. Self-contained on purpose: the parent module imports torch,
which is not present in this venv, and D2 keeps everything inside prime_intellect/.
"""
import argparse
import base64
import json
import os
import re
import sys

SYSTEM_PROMPT = (
    "You are a surgical VQA assistant. Answer the question using the image. "
    "Keep the answer short and in the requested format."
)

FRAME_ROOTS = {"train": ("First_Train_Frames", "Second_Train_Frames"),
               "test": ("First_Test_Frames", "Second_Test_Frames")}
_FRAME_RE = re.compile(r"frame_(\d{2})(\d{2})(\d{2})\.jpg$")


def _ts_seconds(ts):
    h, m, s = (int(x) for x in str(ts).split(":"))
    return h * 3600 + m * 60 + s


class FrameResolver:
    """Maps (video, timestamp_start) -> local frame JPEG (parity with parent repo)."""

    def __init__(self, data_root, split, max_drift):
        self.max_drift = max_drift
        self.index = {}
        for root in FRAME_ROOTS[split]:
            base = os.path.join(data_root, root)
            if not os.path.isdir(base):
                continue
            for vid in os.listdir(base):
                vdir = os.path.join(base, vid)
                if not os.path.isdir(vdir):
                    continue
                table = self.index.setdefault(vid, {})
                for fn in os.listdir(vdir):
                    m = _FRAME_RE.match(fn)
                    if m:
                        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                        table[secs] = os.path.join(vdir, fn)

    def resolve(self, video, timestamp):
        vid = re.sub(r"\.(avi|mp4|mov|mkv)$", "", str(video), flags=re.I)
        table = self.index.get(vid)
        if not table:
            return None, None
        t = _ts_seconds(timestamp)
        if t in table:
            return table[t], 0
        nearest = min(table, key=lambda s: abs(s - t))
        drift = abs(nearest - t)
        return (table[nearest], drift) if drift <= self.max_drift else (None, None)


def load_rows(path, data_root, exclude_source="heico_old",
              max_frame_drift=5, limit=None):
    """Read parquet, resolve frames, return rows with image_path/question/answer."""
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(f"[data] {path} not found")
    df = pd.read_parquet(path)
    excl = {s.strip() for s in exclude_source.split(",") if s.strip()}
    if excl and "source" in df.columns:
        before = len(df)
        df = df[~df["source"].isin(excl)]
        print(f"[data] {path}: filtered source {sorted(excl)} -> {len(df)} rows "
              f"(dropped {before - len(df)})")
    rows = df.to_dict("records")

    split = "train" if "train" in os.path.basename(path).lower() else "test"
    resolver = FrameResolver(data_root, split, max_frame_drift)

    kept, dropped = [], 0
    for r in rows:
        r["video"] = str(r.get("video", ""))
        r["ground_truth"] = str(r.get("ground_truth", r.get("answer", ""))).strip()
        r["answer_format"] = r.get("answer_format", "open_ended")
        r["procedure_type"] = str(r.get("procedure_type", ""))
        img = r.get("local_image") or r.get("image_path")
        drift = r.get("frame_drift", 0)
        if not (img and os.path.exists(img)) and r.get("timestamp_start") is not None:
            img, drift = resolver.resolve(r["video"], r["timestamp_start"])
        if not img:
            dropped += 1
            continue
        r["image_path"], r["frame_drift"] = img, drift
        kept.append(r)
    if dropped:
        print(f"[data] {path}: {len(kept)} rows resolved, {dropped} dropped "
              f"(no frame within {max_frame_drift}s)")
    return kept[:limit] if limit else kept


def image_b64(src):
    with open(src, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")


def build(mode, s):
    user_content = [
        {"type": "image", "image": image_b64(s["image_path"])},
        {"type": "text", "text": s["question"]},
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if mode == "chat":
        messages.append({"role": "assistant", "content": str(s["ground_truth"])})
    rec = {
        "messages": messages,
        "answer_format": s.get("answer_format"),
        "question": s.get("question"),
        "video": s.get("video"),
    }
    if mode == "rl":
        rec["ground_truth"] = {
            "answer": str(s["ground_truth"]),
            "answer_format": s.get("answer_format"),
            "choices": [],
        }
        rec["input_metadata"] = {}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True, help="dir containing First/Second_*_Frames")
    ap.add_argument("--train", required=True, help="frames_train_true.parquet")
    ap.add_argument("--test", required=True, help="frames_test_true.parquet")
    ap.add_argument("--out", default="./orena", help="output dir")
    ap.add_argument("--mode", choices=["chat", "rl"], default="chat")
    ap.add_argument("--max_train_rows", type=int, default=0, help="subset train rows (0=all)")
    ap.add_argument("--max_test_rows", type=int, default=0, help="subset test rows (0=all)")
    ap.add_argument("--max_frame_drift", type=int, default=5)
    ap.add_argument("--exclude_source", default="heico_old")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for split, src, cap in (("train", args.train, args.max_train_rows),
                            ("test", args.test, args.max_test_rows)):
        rows = load_rows(src, args.data_root,
                         exclude_source=args.exclude_source,
                         max_frame_drift=args.max_frame_drift,
                         limit=cap or None)
        out_jsonl = os.path.join(args.out, f"{split}.jsonl")
        with open(out_jsonl, "w") as f:
            for s in rows:
                f.write(json.dumps(build(args.mode, s), ensure_ascii=False) + "\n")
        print(f"{split}: wrote {len(rows)} rows -> {out_jsonl}")
    print("done (mode=%s)" % args.mode)


if __name__ == "__main__":
    main()
