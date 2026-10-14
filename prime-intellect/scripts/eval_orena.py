"""Greedy-decode held-out eval for the ORena checkpoints (Exp 1 SFT, Exp 2 RL).

Same job as eval_goemotions.py but for the Qwen/Qwen3.5-9B VLM: load the base
model, manually merge the separately-saved LoRA adapter (prime-rl does NOT merge
it), then greedy-decode each (image + question) row and score it.

Metric (parity with FIREWORKS_4EXP_CONFIG.md §8.1): reward = exact match after
normalization (number rows tolerate ±0.01); the report metric is the **4-bucket
unweighted meanAccuracy** over {fo_class, number, open_ended, multiple_choice}
(binary excluded). We also print overall exact_match.

Usage (on the pod, prime_rl venv):
  /app/.venv/bin/python eval_orena.py \
      --weights /workspace/outputs/orena-sft-exp1/weights/step_128 \
      --adapter .../lora_adapters/adapter_model.safetensors \
      --data /workspace/data/orena/chat/test.jsonl \
      --lora-r 64 --lora-alpha 128 --out eval_dev_merged.json

⚠️ VLM specifics (verify once on the pod before trusting results):
  - image-content key expected by the Qwen3.5 renderer ({"type":"image"} vs
    {"type":"image_url"}); this script decodes the base64 to a PIL Image and
    passes {"type":"image", "image": PIL}, which is Qwen3.5's native template form.
  - the exact AutoModel class for Qwen3.5-9B on the pod's transformers 5.6.2.
  - LoRA target modules for the vision tower: the merge is key-generic and should
    cover both model.layers.* and model.visual.* (it strips the PEFT wrapper
    prefix and maps .lora_A/.lora_B -> the base weight).
"""
import argparse
import base64
import io
import json
import re
import torch
from safetensors import safe_open
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

SYSTEM_PROMPT = (
    "You are a surgical VQA assistant. Answer the question using the image. "
    "Keep the answer short and in the requested format."
)


def normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip(".,;:!?\"'` ")


def number_ok(pred: str, gold: str, tol: float = 0.01) -> bool:
    try:
        p = float(re.sub(r"[^0-9.\-]", "", pred or "") or "nan")
        g = float(re.sub(r"[^0-9.\-]", "", gold or "") or "nan")
        return abs(p - g) <= tol
    except ValueError:
        return False


def score_row(pred_text: str, gt: dict) -> bool:
    fmt = gt.get("answer_format", "open_ended")
    gold = normalize(str(gt.get("answer", "")))
    pred = normalize(pred_text)
    if fmt == "number":
        return number_ok(pred, gold)
    return pred == gold


def extract_answer(gen: str) -> str:
    if "</think>" in gen:
        gen = gen.split("</think>")[-1]
    # VLM rows in the parent repo also use an "Answer: <value>" tail; prefer it.
    m = re.search(r"Answer:\s*(.*?)\s*$", gen, flags=re.S)
    if m:
        return m.group(1).strip()
    return gen.strip()


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


def load_model(weights_dir):
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(weights_dir, dtype=torch.bfloat16)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.bfloat16)


def build_inputs(processor, pil, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": question},
        ]},
    ]
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return processor(text=[text], images=[pil], return_tensors="pt")
    except Exception:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=[pil], return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--out", default="eval_dev_merged.json")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    scaling = args.lora_alpha / args.lora_r
    print(f"Loading base model from {args.weights}", flush=True)
    model = load_model(args.weights)
    processor = AutoProcessor.from_pretrained(args.weights)

    n = merge_adapter(model, args.adapter, scaling)
    print(f"Merged {n} LoRA layers (scaling={scaling})", flush=True)

    model = model.to("cuda")
    model.eval()

    rows = [json.loads(l) for l in open(args.data)]
    bucket_correct = {}   # answer_format -> (correct, total) for the 4-bucket metric
    total_correct = total = 0
    predictions = []

    for r in rows:
        msgs = r["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        image_part = next(p for p in user["content"] if p["type"] in ("image", "image_url"))
        text_part = next(p for p in user["content"] if p["type"] == "text")
        question = text_part["text"]

        raw_img = image_part.get("image") or image_part.get("image_url", {}).get("url", "")
        if raw_img.startswith("data:"):
            raw_img = raw_img.split(",", 1)[1]
        pil = Image.open(io.BytesIO(base64.b64decode(raw_img))).convert("RGB")

        gt = r.get("ground_truth") or {
            "answer": next(m["content"] for m in msgs if m["role"] == "assistant"),
            "answer_format": r.get("answer_format", "open_ended"),
        }

        inputs = build_inputs(processor, pil, question)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        gen = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gen = extract_answer(gen)
        ok = score_row(gen, gt)
        fmt = gt.get("answer_format", "open_ended")
        c, t = bucket_correct.get(fmt, (0, 0))
        bucket_correct[fmt] = (c + (1 if ok else 0), t + 1)
        total_correct += ok
        total += 1
        predictions.append({"question": question, "gold": gt.get("answer"),
                            "pred": gen, "format": fmt, "correct": bool(ok)})

    buckets = ["fo_class", "number", "open_ended", "multiple_choice"]  # binary excluded
    accs = {b: (bucket_correct[b][0] / bucket_correct[b][1])
            for b in buckets if bucket_correct.get(b, (0, 0))[1] > 0}
    mean_acc_4bucket = round(sum(accs.values()) / len(accs), 4) if accs else None
    res = {
        "n": total,
        "exact_match": round(total_correct / total, 4) if total else None,
        "mean_accuracy_4bucket": mean_acc_4bucket,
        "per_format_accuracy": accs,
        "bucket_totals": {k: v[1] for k, v in bucket_correct.items()},
    }
    json.dump({"metrics": res, "predictions": predictions}, open(args.out, "w"), indent=2)
    print("RESULT_JSON=" + json.dumps(res), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
