"""Greedy-decode eval for the GoEmotions SFT checkpoint (Exp 3).

prime-rl (this pod image) saves the LoRA adapter separately (when
save_adapter_separately=true) but does NOT merge it into the base weights, so we
merge it here manually:  W' = W + (lora_B @ lora_A) * (alpha / r).

Metric definitions (parity with rewards.py reward_goemotions):
  micro-F1 (global)   = 2*sum|pred & gold| / sum(|pred| + |gold|)
  micro-Jaccard       = sum|pred & gold| / sum|pred | gold|
  mean_instance_f1    = mean over rows of 2*|pred&gold| / (|pred| + |gold|)

Run on the pod with the prime_rl venv:
  /app/.venv/bin/python /workspace/eval_goemotions.py
"""
import json
import re
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = "/workspace/outputs/goemotions-sft-exp3/weights/step_100"
ADAPTER = WEIGHTS_DIR + "/lora_adapters/adapter_model.safetensors"
DEV = "/workspace/data/goemotions/chat/dev.jsonl"
OUT = WEIGHTS_DIR + "/eval_dev_merged.json"

LORA_ALPHA = 32.0
LORA_R = 16
SCALING = LORA_ALPHA / LORA_R


def parse_labels(text: str) -> set:
    out = set()
    for tok in re.split(r"[,\n;|]+", text or ""):
        tok = tok.strip().lower().strip(".,;:!?\"'` ")
        if tok:
            out.add(tok)
    return out


def build_prompt(tok, messages) -> str:
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)


def extract_answer(gen: str) -> str:
    if "</think>" in gen:
        gen = gen.split("</think>")[-1]
    return gen


def merge_adapter(model, adapter_path):
    with safe_open(adapter_path, framework="pt") as s:
        adapter = {k: s.get_tensor(k) for k in s.keys()}
    state = model.state_dict()
    merged = 0
    for k, v in adapter.items():
        if not k.endswith("lora_A.weight"):
            continue
        b_key = k.replace("lora_A.weight", "lora_B.weight")
        A = adapter[k].float()
        B = adapter[b_key].float()
        delta = (B @ A) * SCALING
        base_key = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
        assert base_key in state, f"missing base key {base_key}"
        assert delta.shape == state[base_key].shape, (
            f"shape mismatch {base_key}: {tuple(delta.shape)} vs {tuple(state[base_key].shape)}"
        )
        state[base_key] = (state[base_key].float() + delta).to(torch.bfloat16)
        merged += 1
    model.load_state_dict(state)
    return merged


def main():
    print(f"Loading base model from {WEIGHTS_DIR}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    n_merged = merge_adapter(model, ADAPTER)
    print(f"Merged {n_merged} LoRA layers (scaling={SCALING})", flush=True)

    model = model.to("cuda")
    model.eval()

    rows = [json.loads(line) for line in open(DEV)]
    tp = fp = fn = 0
    sum_f1 = sum_jac = 0.0
    n = 0
    predictions = []

    for r in rows:
        msgs = r["messages"]
        sys_usr = [m for m in msgs if m["role"] in ("system", "user")]
        gold_txt = next(m["content"] for m in msgs if m["role"] == "assistant")
        gold = parse_labels(gold_txt)

        prompt = build_prompt(tok, sys_usr)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id
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
        n += 1

        predictions.append({
            "text": next((m["content"] for m in msgs if m["role"] == "user"), ""),
            "gold": sorted(gold),
            "pred": sorted(pred),
            "raw": gen[:160],
        })

    micro_f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    micro_jac = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    res = {
        "n": n,
        "micro_f1": round(micro_f1, 4),
        "jaccard": round(micro_jac, 4),
        "mean_instance_f1": round(sum_f1 / n, 4) if n else None,
        "mean_instance_jaccard": round(sum_jac / n, 4) if n else None,
        "tp": tp, "fp": fp, "fn": fn,
    }

    json.dump({"metrics": res, "predictions": predictions}, open(OUT, "w"), indent=2)
    print("RESULT_JSON=" + json.dumps(res), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
