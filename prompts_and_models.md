# Prompts & Models — Fireworks · SageMaker · Prime Intellect

Cross-platform report for the 4-experiment benchmark (ORena SFT/RL, GoEmotions
SFT/RL). Compiled 2026-10-06 from the code and committed data in this repo.

**Bottom line:** there are **2 base models**, but the **prompts are not identical
across the three platforms** — especially ORena, where each platform ended up with
a *different* system prompt (and Fireworks' own config doc disagrees with the data
it actually shipped).

---

## 1. Models

Two base models carry all four experiments, plus one platform-forced substitution
on the ORena RL leg.

### 1.1 Qwen3.5-9B (vision-language) — ORena (Exps 1 & 2)

| Platform | SFT (Exp 1) | RL (Exp 2) |
|---|---|---|
| Fireworks | `accounts/fireworks/models/qwen3p5-9b` | `qwen3-vl-8b-instruct` ⚠️ |
| SageMaker | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-9B` |
| Prime Intellect | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-9B` |

> ⚠️ **Fireworks substitution:** managed RFT on Fireworks cannot RL-tune
> `qwen3p5-9b` ("model is not tunable for reinforcement fine-tuning"), so Exp 2 ran
> on `qwen3-vl-8b-instruct` (`Qwen/Qwen3-VL-8B-Instruct`, 8.7B) instead. SageMaker
> and Prime Intellect RL the 9B model directly. (Prime Intellect's two ORena legs
> are recorded **BLOCKED** — SFT by VLM-SFT version drift, RL by image toolchain
> drift — but the *intended* model is still `Qwen/Qwen3.5-9B`.)

### 1.2 Qwen3-0.6B (text) — GoEmotions (Exps 3 & 4)

| Platform | SFT (Exp 3) | RL (Exp 4) |
|---|---|---|
| Fireworks | `accounts/fireworks/models/qwen3-0p6b` | `qwen3-0p6b` |
| SageMaker | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |
| Prime Intellect | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |

Identical model on all three platforms; no substitution.

### Summary

| | Model | Params | Used for |
|---|---|---|---|
| 1 | `Qwen/Qwen3.5-9B` | 9.4B (VLM) | ORena (SFT everywhere; RL on SageMaker + Prime) |
| 2 | `Qwen/Qwen3-0.6B` | 0.75B (text) | GoEmotions (SFT + RL, all platforms) |
| — | `Qwen/Qwen3-VL-8B-Instruct` | 8.7B (VLM) | ORena RL **on Fireworks only** (forced substitution) |

---

## 2. Prompts

Message structure is identical everywhere (`system` + `user`, plus an `assistant`
gold turn for SFT). The **system-prompt text is not**.

### 2.1 ORena (vision + text) — three different system prompts

Input shape (identical): `system` + `user[ image, text-question ]`
(+ `assistant` reference answer in SFT). Image is base64-inline on Fireworks and
Prime Intellect; a frame-file path on SageMaker.

#### Fireworks — "surgical video analyst" (long; the data it actually shipped)

```
You are a surgical video analyst. You are shown a single frame from a laparoscopic or endoscopic procedure and asked one question about the surgical FOREIGN OBJECTS visible in it.

The only foreign object classes that exist are:
Clip, Sponge, Silicone loop, External drain, Specimen, Specimen bag, Needle, Gallstone.
Never invent a class outside this list. Surgical instruments (graspers, scissors, trocars, cautery hooks) are NOT foreign objects and must never be counted or named.

Work through the frame quadrant by quadrant, noting every candidate object and its position, discarding instruments and anatomy. Keep your reasoning brief.

Your final reply must be ONLY the answer itself — no explanation, no units, no restating of the question, no extra words.

Answer formats:
- "how many ..."          -> a single integer, e.g. 3
- "yes or no"             -> exactly one word: yes or no
- "which class(es) ..."   -> comma-separated class names, e.g. Clip, Sponge  (or: none)
```

Source: `fireworks/reference/orena-focus-rft/export_dataset.py`; confirmed in
`data/orena/orena_{sft,rl}_sample_2rows.jsonl`.

#### Prime Intellect — "surgical VQA assistant" (short)

```
You are a surgical VQA assistant. Answer the question using the image. Keep the answer short and in the requested format.
```

Source: `prime-intellect/scripts/convert_orana_to_hf.py`,
`prime-intellect/scripts/eval_orena.py`.

#### SageMaker — "surgical assistant" + class definitions + format instruction (RL)

```
You are a surgical assistant. You are given an endoscopic frame from a minimally invasive procedure. Analyze the image and answer the surgical question based on the visual evidence. Be precise and concise.

{fo_defs}

Question: {question}
Expected Format: {fmt}
Instruction: {instruction}

Return ONLY the answer. Do NOT explain. Do NOT add any text before or after it.
```

- `{fo_defs}` = `sagemaker/scripts/src/fo_defs.txt` — the 8 canonical foreign-object
  classes with descriptions (Clip, Silicone loop, Needle, Sponge, Specimen bag,
  Specimen, External drain, Gallstone).
- `{instruction}` is per answer-format, e.g. `binary` → "Return only yes or no.",
  `number` → "Return only a non-negative integer.", `fo_class` → "Return only the
  foreign object class name(s), comma-separated.", `multiple_choice` → "Return only
  one option from: top/left, top/right, bottom/left, bottom/right."

Source: `sagemaker/scripts/src/orena_data.py`. (SageMaker's ORena **SFT** reads a
pre-built swift-style `train.jsonl` from S3 with a `<image>` placeholder; its system
prompt is not captured in this repo.)

#### ORena prompt — the discrepancy, stated plainly

| | ORena system prompt |
|---|---|
| Fireworks (data) | "surgical **video analyst**" — long, foreign-object + quadrant reasoning |
| Prime Intellect | "surgical **VQA assistant**" — one line |
| SageMaker (RL) | "surgical **assistant**" — + `fo_defs` + per-format `Instruction` |

Note `fireworks/config/FIREWORKS_4EXP_CONFIG.md` §3.1 documents a *fourth* variant
— the one-line "surgical VQA assistant" — which **does not match the data Fireworks
actually shipped** ("surgical video analyst"). So: Fireworks' docs and data disagree;
Prime Intellect matches Fireworks' *docs*; SageMaker matches *neither*. This is a
real cross-platform comparability caveat for the ORena legs.

### 2.2 GoEmotions (text) — two variants

Input shape (identical): `system` + `user[ text ]`
(+ `assistant` comma-separated label list in SFT).

#### Fireworks — richer (adds "emotion", "and nothing else", explicit label list)

```
Classify the text into one or more of the 28 GoEmotions emotion labels. Reply with a comma-separated list of label names only, and nothing else. Valid labels: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral.
```

Source: `fireworks/scripts/20_build_datasets.py` (`GE_SYS`); confirmed in
`data/goemotions/goemotions_{sft,rl}.jsonl`.

#### Prime Intellect + SageMaker — shorter (identical on both)

```
Classify the text into one or more of the 28 GoEmotions labels. Reply with a comma-separated list of label names only.
```

Source: `prime-intellect/scripts/build_goemotions_dataset.py`,
`sagemaker/scripts/src/goemotions_{sft,rl}.py`. Prime Intellect and SageMaker match
each other and match `FIREWORKS_4EXP_CONFIG.md` §3.2; Fireworks' shipped data is the
only one that adds the "emotion" / "and nothing else" / "Valid labels" enrichment.

---

## 3. References (where each prompt lives in the repo)

| Item | File(s) |
|---|---|
| Fireworks ORena prompt | `fireworks/reference/orena-focus-rft/export_dataset.py`, `data/orena/*_sample_2rows.jsonl` |
| Fireworks GoEmotions prompt | `fireworks/scripts/20_build_datasets.py`, `data/goemotions/*.jsonl` |
| Prime Intellect ORena prompt | `prime-intellect/scripts/convert_orana_to_hf.py`, `eval_orena.py` |
| Prime Intellect GoEmotions prompt | `prime-intellect/scripts/build_goemotions_dataset.py`, `eval_goemotions_rl.py` |
| SageMaker ORena prompt (RL) | `sagemaker/scripts/src/orena_data.py`, `fo_defs.txt` |
| SageMaker GoEmotions prompt | `sagemaker/scripts/src/goemotions_sft.py`, `goemotions_rl.py` |
| Models (actual, per platform) | `{fireworks,sagemaker,prime-intellect}/metrics/final_results.json` |
| Canonical input data | `data/manifest.json`, `data/README.md` |
