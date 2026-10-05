# Decisions & Execution Log — Prime Intellect (ORena + GoEmotions, 4 experiments)

**Purpose:** one living record of every decision and every step from now until we have the four results. Append to it; never rewrite history. Status legend:

- ✅ **Locked** — decided, do not relitigate.
- ⏳ **Open** — not yet decided.
- 🚫 **Blocked** — waiting on the user / an external gate.
- ✔ **Done** — completed.

_Last updated: 2026-10-05._

---

## Decisions (locked)

| # | Decision | Status |
|---|---|---|
| D1 | Reproduce the 4 experiments in `FIREWORKS_4EXP_CONFIG.md` on **Prime Intellect** (not Fireworks). | ✅ |
| D2 | Everything lives in `prime_intellect/` only; nothing touches the parent `med_vlm/` training code or runs. | ✅ |
| D3 | One document per step (`00`–`09`) + scripts; this file is the cross-cutting log. | ✅ |
| D4 | **Small runs only**, each ≤ 20–30 min: `timeout 1800` + a `--max-steps` cap. | ✅ |
| D5 | Evidence per run = **screenshot + loss/reward plot + final result** (`result.json`). | ✅ |
| D6 | **Parity rule:** keep every reference hyperparameter identical; shrink *only* data volume and step count. | ✅ |
| D7 | Base models: `Qwen/Qwen3.5-9B` (VLM) for ORena (Exp 1–2); `Qwen/Qwen3-0.6B` (text) for GoEmotions (Exp 3–4). | ✅ |
| D8 | ORena data: use the repo's parquet + frame JPEGs, convert to **base64-inline JSONL** (parity with §3.1) via `convert_orana_to_hf.py`. | ✅ |
| D9 | GoEmotions data: **user must provide** (`train/dev/test.tsv` + `emotions.txt`); build via `build_goemotions_dataset.py`. | ✅ |
| D10 | Logging = **tensorboard** (self-run) / `prime train metrics` (hosted) / CSV. **No W&B.** | ✅ |
| D11 | ~~Compute = Path B — prime-rl on the user's own GPU box~~ **OVERTURNED 2026-10-02** (user has no own GPU box). Superseded by D16. | ❌ |
| D12 | Secrets set via **masked prompts** (`prime login`, `huggingface-cli login`) or `~/.zshrc`; **never paste raw keys into chat.** | ✅ |
| D13 | With Path B, `PRIME_API_KEY` + `HF_TOKEN` are **not required** for training (prime-rl is open source, local JSONL works); keep them for the `prime env` hub + full-FT fallback. | ✅ |
| D14 | Run order: `01` recon → `02`/`03` data → `04` rewards → **run Exp 3 (GoEmotions SFT) first** to de-risk the platform, then Exp 1 → Exp 2 → Exp 4 → `09` results. | ✅ |
| D15 | RL runs are **fresh** (no warm-start from the SFT adapter), per reference §1. | ✅ |
| D16 | Compute = **PI on-demand pod + self-run prime-rl** (`prime pods create` → `prime pods ssh` → `uv run sft/rl`). Full-FT (c) out; hosted LoRA SFT gated on closed-beta volumes; hosted RL = env-hub fallback. Workhorse = A100 40GB (stable, $1.99/hr); H100 80GB when available. | ✅ |
| D17 | GoEmotions source = **HF `google-research-datasets/go_emotions`** (canonical 43,410/5,426/5,427, 28 labels), fetched 2026-10-02. Kaggle `debarshichanda/goemotions` (user's link) is the alternative but needs a Kaggle API token; the HF canonical data is the reference-faithful source. | ✅ |

### Small-run sizes (D4/D6 applied)

| Exp | Base model | Method | Subset | Group | Step cap |
|---|---|---|---|---|---|
| 1 | Qwen3.5-9B (VLM) | SFT LoRA r=64 | 300 train / 100 eval | — | 128 |
| 2 | Qwen3.5-9B (VLM) | GRPO LoRA r=16 | 128 prompts | 8 | 48 |
| 3 | Qwen3-0.6B (text) | SFT LoRA r=16 | 800 train / 200 dev | — | 256 |
| 4 | Qwen3-0.6B (text) | GRPO LoRA r=16 | 256 prompts | 8 | 64 |

If a run hits the 30-min wall → halve the subset and rerun (record the partial score too).

---

## Steps taken (execution log)

| # | Date | Step | Status |
|---|---|---|---|
| S1 | 2026-10-02 | Read `FIREWORKS_4EXP_CONFIG.md`; surveyed Prime Intellect docs (`prime-rl` training, full-FT beta, `prime` CLI/SDK). Key finding: VLM support is constrained to Qwen3.5 dense/MoE. | ✔ |
| S2 | 2026-10-02 | Created `prime_intellect/` + `STEPS.md` + `convert_orana_to_hf.py`. | ✔ |
| S3 | 2026-10-02 | Wrote per-step docs `00`–`09`, `build_goemotions_dataset.py`, `rewards.py`. | ✔ |
| S4 | 2026-10-02 | Answered key/token provisioning; locked logging (D10) + compute (D11) + secrets policy (D12). | ✔ |
| S5 | 2026-10-02 | Created this `DECISIONS_LOG.md`. | ✔ |
| S6 | 2026-10-02 | Ran `01` recon (auth ✅; `prime train gpus` empty → full-FT out; `Qwen3.5-9B` in hosted catalog, `Qwen3-0.6B` not; on-demand GPUs ≤ A100_80GB). **Path b confirmed.** | ✔ |
| S7 | 2026-10-05 | Data prep (03 + 02): GoEmotions fetched (HF canonical) → chat/rl JSONL (800/200); ORena → base64-inline chat (300/100) + rl (128/100) JSONL (self-contained `convert_orana_to_hf.py`, no torch dep). | ✔ |
| S8 | 2026-10-02 | Run `04` rewards self-test — `python3 rewards.py` → OK (ORena + GoEmotions reward fns). | ✔ |
| S9 | 2026-10-02 | Run Exp 3 (GoEmotions SFT) — de-risk run. **DONE** (pod `234fc42f…`): loss 5.79→0.167, dev micro-F1 0.4956. | ✔ |
| S10 | 2026-10-05 | Run Exp 1 (ORena SFT). **BLOCKED** — the pod's prime-rl 0.7.0 rejects `[model.vlm]` for SFT ("renderer-only SFT does not support VLMs yet"); GitHub main removed the check. Recorded as version drift (see below). | 🚫 |
| S11 | — | Run Exp 2 (ORena GRPO). Feasible on 0.7.0 (the RL config supports `[model.vlm]`; rollout via vLLM). Deferred. | ⏳ |
| S12 | 2026-10-05 | Run Exp 4 (GoEmotions GRPO). **DONE** (pod `484b6053…` A10): 64 steps, final Jaccard 0.0178, held-out micro-F1 0.0349. | ✔ |
| S13 | 2026-10-05 | Run `09` → consolidated into `final_results.json` (canonical schema) + benchmark repo PR. | ✔ |

✔ 2026-10-02 — **Installation:** `uv` already present; installed `prime` 0.9.1 + `hf` (huggingface-hub 2.1.1) via `uv tool install`. Confirmed auth commands: `prime login` / `prime config set-api-key`; HF: `hf auth login`.

✔ 2026-10-02 — **Compute recon (B2):** `prime pods`/`prime images`/`prime volumes` confirmed; `prime availability list` = live on-demand stock (A100 40GB ×4, A10 ×1, H100 80GB volatile). Full-FT still empty; volumes closed-beta. → **D16** (PI pod + self-run prime-rl).

✔ 2026-10-02 — **GoEmotions (B3):** Kaggle CLI absent + no token; fetched canonical `google-research-datasets/go_emotions` (parquet) instead, converted via `convert_goemotions_parquet_to_tsv.py`, built `goemotions/{chat,rl}/{train,dev}.jsonl` (800/200; 28 labels, ids 0–27 verified).

✔ 2026-10-02 — **SSH key diagnosis + fix (root cause found):** pod `0c22b573…` key injection was actually **working** — verbose ssh showed `Server accepts key: …SHA256:3evIu9mlfnw4saD3yNrMma+/sT1cQABOXVldfZBmpJY` (public key present in authorized_keys), but the local `~/.ssh/id_ed25519` is **passphrase-protected**, so non-interactive ssh couldn't sign (`we did not send a packet, disable method` → "Permission denied"). Fix: generated passphrase-less `~/.ssh/pi_id_ed25519`, registered as `pi-pod-key` (API auto-promoted it `isPrimary: true`, demoted `holbox-macbook`), `prime config set-ssh-key-path`, terminated `0c22b573…` and recreated `234fc42f…` (A100 40GB, us-east-1). No raw keys pasted.

✔ 2026-10-02 — **prime-rl on-pod discovery + config fixes (headless gotchas):** prime-rl lives at **`/app`** (not `/workspace/prime-rl`). The pod image ships an **older prime-rl** than GitHub main → `[run]` section is rejected (`Extra inputs are not permitted`); the run name lives in `output_dir` instead. `[data] name` is passed to `datasets.load_dataset(name, subset, split)` → a bare `.jsonl` **file** path fails (`Couldn't find any data file`); must point `name` at the **directory** containing `train.jsonl`. Configs rewritten accordingly (`output_dir=/workspace/outputs/…`, `name=/workspace/data/goemotions/chat`).

✔ 2026-10-02 — **Smoke SFT passed** (fake data, `Qwen/Qwen3-0.6B`, 10 steps, A100 40GB): config validated, model downloaded, LoRA+optimizer+scheduler+training loop all work; peak 11.1 GiB, ~936 tok/s. → **Exp 3 launched** (GoEmotions SFT, 100 steps, LoRA r=16, batch 64/µ8, cosine warmup 10). Observed learning: loss 5.79 → 3.37 over steps 1–6; `Qwen3Renderer` auto-selected; LoRA = 10.1M/440M params; peak 3.3 GiB.

✔ 2026-10-03 — **prime-rl LoRA checkpoint bug (root-caused + worked around).** Default `save_adapter_separately=false` makes `clean_lora_state_dict` **strip** `lora_A`/`lora_B` and keep only the frozen base weights — the docstring claims "before merging into full model weights" but no merge happens. So the saved `model.safetensors` is the *base* model and the fine-tune is silently lost (first eval = base behavior, micro-F1 0.02). Fix: `[ckpt.weights] save_adapter_separately = true` → re-run → `lora_adapters/adapter_model.safetensors` (PEFT-format, 392 keys) saved. Eval merges it manually: `W' = W + (lora_B@lora_A)·α/r` (α/r = 32/16 = 2.0, confirmed via `set_multilora_scaling` docstring "alpha/rank").

✔ 2026-10-03 — **Exp 3 COMPLETE (GoEmotions SFT).** Final loss **0.167** (step 100). Held-out dev (200 rows, greedy decode, thinking disabled): **micro-F1 0.4956**, micro-Jaccard 0.3294, mean instance-F1 0.4948 (TP/FP/FN = 113/104/126). `eval_goemotions.py` (local + on-pod) does the merge + decode. Result recorded in `EXPERIMENT_RESULTS.md` + `fireworks-vs-sagemaker-comparison.md`.

✔ 2026-10-05 — **ORena data built (step 02) + remaining configs & eval scripts.** Rewrote `convert_orana_to_hf.py` self-contained (parent `train_qwen3vl` imports `torch`, which is not in the local venv; D2 = keep everything in `prime_intellect/`). Built `orena/chat/{train,test}.jsonl` (300/100, base64-inline JPEG) and `orena/rl/{train,test}.jsonl` (128/100). Format mix verified (fo_class/number/open_ended/multiple_choice/binary; binary excluded from the 4-bucket metric). Wrote `configs/sft_orena.toml` (Exp 1), `configs/rl_orena.toml` (Exp 2), `configs/rl_goemotions.toml` (Exp 4), and `eval_orena.py` (VLM held-out eval with key-generic LoRA merge + 4-bucket meanAccuracy). RL configs use the GitHub-main `[orchestrator]` schema and are flagged **not verified** against the pod's older prime-rl (which already rejected `[run]`).

✔ 2026-10-05 — **Exp 1 (ORena SFT) BLOCKED — VLM SFT version drift.** The pod's prime-rl 0.7.0 SFT path is renderer-only and raises `validate_renderer_vs_vlm`: "renderer-only SFT does not support VLMs yet. The renderer tokenizes text-only message dicts client-side and cannot handle image inputs." GitHub main has **removed** this validator (its only remaining VLM/SFT constraint is `freeze_vision_encoder=false` being LoRA-incompatible), so this is a pod-image version gap, not a hard platform limit. The mirror-image of Fireworks' gap: Fireworks can't RL `qwen3p5-9b`; Prime Intellect 0.7.0 can't SFT a VLM but **can** RL one (`rl.py:109` carries `vlm`, rollout through vLLM). The H200 provisioned for Exp 1/2 was **terminated** (idle at $4.50/hr, $15.51 wasted) rather than attempt a risky prime-rl upgrade.

✔ 2026-10-05 — **Exp 4 COMPLETE (GoEmotions GRPO).** pod `484b6053…` (A10 24GB), 64 steps, fresh base model, reward = multi-label Jaccard, `enable_thinking=false`. Final held-out dev (200 rows, greedy): **micro-F1 0.0349, Jaccard 0.0178**; `jaccard_curve` flat/noisy (0–0.073). A fresh 0.6B + Jaccard over 64 steps is not enough signal vs Fireworks' 0.175→0.377 on 5000 prompts. **Three launch fixes:** (a) `CUDA_VISIBLE_DEVICES="0,0"` for the single-GPU disjoint-GPU check; (b) `[inference] gpu_memory_utilization=0.6` + `[inference.model] max_model_len=512` (vLLM otherwise fails: trainer holds ~3.8 GiB); (c) `enable_thinking=false` (default thinking ate the 64-token budget → Truncation 100%) + `post_batch_filters` `enforce=false` (zero_advantage dropped every group → "10 consecutive empty train batches").

## Recon findings (01, 2026-10-02)

- Auth ✅ — PI `purchase@holbox.ai` (token scopes: `rft`, `environments`, `instances` RW), HF `Sheel8Holbox`.
- `prime train gpus` → `{"gpuTypes": []}` → **full-FT (path c) not dispatchable** for this account.
- Hosted trainable models include `Qwen/Qwen3.5-9B` ✅ (ORena base) but **not** `Qwen/Qwen3-0.6B` (GoEmotions base). Hosted-only substitute would be `Qwen3.5-0.8B` or `Llama-3.2-1B` — **not needed on path b**, where we load any HF model directly.
- On-demand GPUs: A10 / A100(40/80 GB) / A6000 / L40 / L40S / RTX6000Ada / H100. **Live stock 2026-10-02:** A100 40GB ×4, A10 24GB ×1, H100 80GB ×1 (volatile — listed then sold within ~2 min); the `A100_80GB` filter returned empty at that moment.
- Hosted SFT needs a named **volume** (closed beta) for real data ("without `--volume`, only fake datasets work"); hosted RL uses `env[]` environments; config = prime-rl schema (`loss` default `rl`, `rollouts_per_example` 8, `checkpoint_id` for warm-start).
- **Conclusion:** full-FT (c) out; hosted LoRA SFT gated on closed-beta volumes; hosted RL = env-hub fallback. **On-demand pods are rentable** (`prime pods create` → `prime pods ssh`) → **D16: self-run prime-rl on a PI pod** (A100 40GB workhorse, H100 80GB when it shows).
- Remaining gate (B4): confirm prime-rl registers `Qwen3.5-9B` as a VLM class (code-level check), and `Qwen3-0.6B`/`Qwen3` text renderer.

---

## Blocked on (user / external gates)

| # | Item | Owner | Status |
|---|---|---|---|
| B1 | Authenticate: run `prime login` (or `prime config set-api-key`) + `hf auth login` — masked. CLIs already installed. | user | ✔ |
| B2 | ~~SSH access to the GPU box~~ → user has no own GPU box; compute is a PI on-demand pod (D16). | user | ✔ |
| B3 | GoEmotions data — fetched from HF `google-research-datasets/go_emotions` (canonical, zero-auth) 2026-10-02, built to JSONL. Kaggle `debarshichanda/goemotions` (user's link) needs a Kaggle API token if we want that exact file instead. | user | ✔ |
| B4 | Confirm `Qwen3.5-9B` is VLM-trainable on prime-rl. ✅ Resolved: prime-rl registers "Qwen3/Qwen3.5 VLMs" (`qwen3_vl`, `qwen3_5` config types); `Qwen3-0.6B` text is their getting-started model. Exact enable flag = `[model] impl="custom"` + `[model.vlm]` (verify in `docs/advanced.md#multimodal-training` at run time). | recon | ✔ |
| B5 | ~~Prime Intellect API key unauthorized (2026-10-05)~~ **RESOLVED** — a new key was generated and `prime login` re-run (masked). Work resumed on pods `484b6053…` (A10, Exp 4) and `19b97318…` (H200, provisioned for ORena, terminated idle). | user | ✔ |

---

## Results (empty until we have them)

| Exp | Metric | Fireworks (ref) | Prime Intellect | Δ |
|---|---|---|---|---|
| 1 ORena SFT | meanAccuracy (4-bucket) | — | 🚫 blocked (VLM SFT version drift) | — |
| 2 ORena RL | meanAccuracy (4-bucket) | — | ⏳ not started (feasible) | — |
| 3 GoEmotions SFT | micro-F1 | 0.313 (train loss) | **0.4956** | 800 vs 43k samples |
| 4 GoEmotions RL | micro-F1 / Jaccard | 0.175→0.377 (F1) | **0.0349** / 0.0178 | fresh 0.6B + Jaccard over 64 steps ≈ no signal |

---

## Append convention

When a decision is made or a step completes, update this file in the same commit/turn:
- flip a `⏳`/`🚫` to `✔`,
- add a dated row to the step log,
- add the number to the results table when a score lands.

Do not delete resolved rows — the point is the full trail from now to results.
