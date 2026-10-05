"""Greedy-decode held-out eval for the GoEmotions RL checkpoint (Exp 4).

Same merge-then-decode job as eval_goemotions.py, but for the RL JSONL format:
messages = system + user (no assistant), and ground_truth.labels holds the gold
set. Scores with reward_goemotions (Jaccard is the reward; micro-F1 tracked).

Usage (on the pod):
  /app/.venv/bin/python eval_goemotions_rl.py \
      --weights /workspace/outputs/goemotions-rl-exp4/weights/step_64 \
      --adapter .../lora_adapters/adapter_model.safetensors \
      --data /workspace/data/goemotions/rl/dev.jsonl \
      --lora-r 16 --lora-alpha 32 --out eval_dev_merged.json
"""
import argparse
import json
import re
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "Classify the text into one or more of the 28 GoEmotions labels. "
    "Reply with a comma-separated list of label names only."
)


def parse_labels(text: str) -> set:
    out = set()
    for tok in re.split(r"[,\n;|]+", text or ""):
        tok = tok.strip().lower().strip(".,;:!?\"'` ")
        if tok:
            out.add(tok)
    return out


def build_prompt(tok, messages) -> str:
    sys_usr = [m for m in messages if m["role"] in ("system", "user")]
    try:
        return tok.apply_chat_template(
            sys_usr, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        parts = []
        for m in sys_usr:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)


def extract_answer(gen: str) -> str:
    if "</think>" in gen:
        gen = gen.split("</think>")[-1]
    return gen


def merge_adapter(model, adapter_path, scaling):
    with safe_open(adapter_path, framework="pt") as s:
        adapter = {k: s.get_tensor(k) for k in s.keys()}
    state = model.state_dict()
    merged = 0
    for k in adapter:
        if not k.endswith("lora_A.weight"):
            continue
        b_key = k.replace("lora_A.weight", "lora_B.weight")
        A = adapter[k].float()
        B = adapter[b_key].float()
        delta = (B @ A) * scaling
        base_key = None
        for prefix in ("base_model.model.", "base_model.model.model.", ""):
            cand = k.replace(prefix, "", 1).replace(".lora_A.weight", ".weight")
            if cand in state and state[cand].shape == delta.shape:
                base_key = cand
                break
        assert base_key is not None, f"no base weight for adapter key {k}"
        state[base_key] = (state[base_key].float() + delta).to(state[base_key].dtype)
        merged += 1
    model.load_state_dict(state)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--out", default="eval_dev_merged.json")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    scaling = args.lora_alpha / args.lora_r
    print(f"Loading base model from {args.weights}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.weights, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.weights)

    n = merge_adapter(model, args.adapter, scaling)
    print(f"Merged {n} LoRA layers (scaling={scaling})", flush=True)

    model = model.to("cuda")
    model.eval()

    rows = [json.loads(l) for l in open(args.data)]
    tp = fp = fn = 0
    sum_f1 = sum_jac = 0.0
    predictions = []

    for r in rows:
        msgs = r["messages"]
        gold = set(r["ground_truth"]["labels"])
        prompt = build_prompt(tok, msgs)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gen = extract_answer(gen)
        pred = parse_labels(gen)

        inter = pred & gold
        tp += len(inter)
        fp += len(pred - gold)
        fn += len(gold - pred)
        f1 = 2 * len(inter) / (len(pred) + len(gold)) if (pred or gold) else 1.0
        jac = len(inter) / len(pred | gold) if (pred | gold) else 1.0
        sum_f1 += f1
        sum_jac += jac

        predictions.append({
            "text": next((m["content"] for m in msgs if m["role"] == "user"), ""),
            "gold": sorted(gold), "pred": sorted(pred), "raw": gen[:160],
        })

    micro_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    micro_jac = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    res = {
        "n": len(rows),
        "micro_f1": round(micro_f1, 4),
        "jaccard": round(micro_jac, 4),
        "mean_instance_f1": round(sum_f1 / len(rows), 4) if rows else None,
        "mean_instance_jaccard": round(sum_jac / len(rows), 4) if rows else None,
        "tp": tp, "fp": fp, "fn": fn,
    }
    json.dump({"metrics": res, "predictions": predictions}, open(args.out, "w"), indent=2)
    print("RESULT_JSON=" + json.dumps(res), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
