#!/usr/bin/env python3
"""CholecT50 SFT on Thinking Machines Tinker.

Fine-tunes Qwen3.5-4B (vision-language, LoRA) to emit surgical action triplets
("instrument-verb-target") for a given endoscopic frame. Uses Tinker's low-level
training loop: build a Datum per (image, prompt, answer), then
forward_backward(loss_fn="cross_entropy") + optim_step on managed GPUs.

The actual model weights never run locally -- only the tokenizer + image
processor (small) are loaded here. Auth is read from TINKER_API_KEY.

Memory: datums are built one micro-batch at a time (the frames are already on
disk), so the local process holds only ~1 batch + the tokenizer, not the whole
dataset. This matters because the host has 8 GB RAM and macOS otherwise kills
the orchestrator under memory pressure.

The run checkpoints the LoRA training state (save_state -> `weights/`) every
--save_every steps and is resumable: if the process is killed, relaunch the same
command and it continues from the last saved step via
create_training_client_from_state. The final `weights/` checkpoint is also what
tinker_rl.py loads to start RL, and what tinker_eval.py loads for inference.
Pass --fresh to ignore a saved state.

Smoke test:
    TINKER_API_KEY=... python tinker_sft.py --max_samples 4 --max_steps 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from tinker import AdamParams, ServiceClient
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import compute_mean_nll, conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cholec_data import download_frames, load_manifest
from reward import PROMPT

MODEL = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"  # no <think> blocks -> plain triplet output
MAX_LENGTH = 8192


def build_datum(row, path, tokenizer, imgproc, renderer, max_length=MAX_LENGTH):
    img = Image.open(path).convert("RGB")
    conv = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]},
        {"role": "assistant", "content": ", ".join(row["triplets"])},
    ]
    return conversation_to_datum(
        conv, renderer, max_length=max_length,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)


def save_checkpoint(client, step, args, name=None):
    """Persist the LoRA training state and remember where to resume from.

    save_state() produces a `weights/` path, which is the only kind that
    create_training_client_from_state (SFT resume *and* RL init) can load.
    save_weights_for_sampler() produces `sampler_weights/`, which cannot be
    resumed from -- only used for inference/sampling.
    """
    name = name or f"cholec-sft-{step + 1}"
    print(f"saving checkpoint {name} (step {step + 1}) ...", flush=True)
    ckpt = client.save_state(name=name).result()
    print(f"  checkpoint path: {ckpt.path}", flush=True)
    with open(args.ckpt_file, "w") as f:
        f.write(ckpt.path + "\n")
    with open(args.state_file, "w") as f:
        json.dump({"step": step, "ckpt": ckpt.path}, f)
    return ckpt.path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all train rows")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = one epoch")
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--cache_dir", type=str, default="/tmp/cholec_frames")
    ap.add_argument("--log_path", type=str, default="/tmp/cholec_tinker_sft.jsonl")
    ap.add_argument("--save_name", type=str, default="cholec-sft")
    ap.add_argument("--save_every", type=int, default=10, help="checkpoint every N steps")
    ap.add_argument("--ckpt_file", type=str, default="/tmp/cholec_tinker_sft_ckpt.txt")
    ap.add_argument("--state_file", type=str, default="/tmp/cholec_tinker_sft_state.json")
    ap.add_argument("--fresh", action="store_true", help="ignore any saved state, start over")
    args = ap.parse_args()

    t0 = time.time()
    print("loading tokenizer + image processor ...", flush=True)
    tokenizer = get_tokenizer(MODEL)
    imgproc = get_image_processor(MODEL)
    renderer = get_renderer(RENDERER_NAME, tokenizer, image_processor=imgproc, model_name=MODEL)
    print(f"  tokenizer+processor ready in {time.time()-t0:.1f}s", flush=True)

    rows = load_manifest("train")
    if args.max_samples:
        rows = rows[: args.max_samples]
    members = [r["member"] for r in rows]
    paths = download_frames(members, args.cache_dir)  # idempotent; cached on disk
    rows = [r for r in rows if paths.get(r["member"])]  # drop unresolved members
    print(f"resolved {len(rows)} rows in {time.time()-t0:.1f}s", flush=True)

    sc = ServiceClient()

    # Resume from a saved state unless told otherwise.
    start_step = 0
    client = None
    if not args.fresh and os.path.exists(args.state_file):
        try:
            state = json.load(open(args.state_file))
            if state.get("ckpt"):
                print(f"resuming from checkpoint at step {state['step'] + 1}: {state['ckpt']}", flush=True)
                t1 = time.time()
                client = sc.create_training_client_from_state(state["ckpt"])
                start_step = state["step"] + 1
                print(f"  training client resumed in {time.time()-t1:.1f}s", flush=True)
        except Exception as e:
            print(f"  could not resume ({e}); starting fresh", flush=True)
            start_step = 0
            client = None

    if client is None:
        t1 = time.time()
        print("creating LoRA training client (model load on Tinker side) ...", flush=True)
        client = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
        print(f"  training client ready in {time.time()-t1:.1f}s", flush=True)

    mb = args.micro_batch
    n_batches = (len(rows) + mb - 1) // mb
    n_steps = n_batches if args.max_steps == 0 else min(args.max_steps, n_batches)
    adam = AdamParams(learning_rate=args.lr)

    logf = open(args.log_path, "a")
    total_tokens = 0
    for step in range(start_step, n_steps):
        # Build just this micro-batch's datums (frames already on disk).
        batch = []
        for r in rows[step * mb: (step + 1) * mb]:
            p = paths[r["member"]]
            batch.append(build_datum(r, p, tokenizer, imgproc, renderer))
        bt = sum(d.model_input.length for d in batch)
        total_tokens += bt

        s = time.time()
        try:
            fwd = client.forward_backward(batch, loss_fn="cross_entropy")
            out = fwd.result(timeout=120)
            opt = client.optim_step(adam_params=adam)
            opt_res = opt.result(timeout=120)
        except Exception as e:
            print(f"step {step+1} failed: {e!r}; exiting (resume from checkpoint)", flush=True)
            logf.close()
            raise SystemExit(1)
        dt = time.time() - s

        logprobs = [x["logprobs"] for x in out.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in batch]
        nll = compute_mean_nll(logprobs, weights)
        del batch  # free the datum tokens before the next iteration

        rec = {"step": step, "loss": nll, "batch_tokens": bt,
               "step_seconds": round(dt, 2), "wall_s": round(time.time() - t1, 1)}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"step {step+1}/{n_steps}: loss={nll:.4f} tokens={bt} "
              f"({dt:.1f}s/step, {bt/dt:.0f} tok/s)", flush=True)

        if args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(client, step, args)

    logf.close()
    print(f"DONE: {n_steps} steps (started at {start_step}), {total_tokens} tokens, "
          f"total wall {time.time()-t1:.1f}s", flush=True)

    print("saving final SFT checkpoint (training state) ...", flush=True)
    save_checkpoint(client, n_steps - 1, args, name="cholec-sft-final")
    print(f"  wrote checkpoint path to {args.ckpt_file}", flush=True)


if __name__ == "__main__":
    main()
