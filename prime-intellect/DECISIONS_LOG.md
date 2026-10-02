# Decisions & Execution Log — Prime Intellect (ORena + GoEmotions, 4 experiments)

**Purpose:** one living record of every decision and every step from now until we have the four results. Append to it; never rewrite history. Status legend:

- ✅ **Locked** — decided, do not relitigate.
- ⏳ **Open** — not yet decided.
- 🚫 **Blocked** — waiting on the user / an external gate.
- ✔ **Done** — completed.

_Last updated: 2026-10-02._

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
| S7 | 2026-10-02 | Data prep (03): GoEmotions fetched (HF canonical) → parquet→tsv → chat/rl JSONL built (800/200). ORena JSONL (02) still pending. | ⏳ |
| S8 | 2026-10-02 | Run `04` rewards self-test — `python3 rewards.py` → OK (ORena + GoEmotions reward fns). | ✔ |
| S9 | 2026-10-02 | Run Exp 3 (GoEmotions SFT) — de-risk run. **IN PROGRESS** (pod `234fc42f…`). | ⏳ |
| S10 | — | Run Exp 1 (ORena SFT). | ⏳ |
| S11 | — | Run Exp 2 (ORena GRPO). | ⏳ |
| S12 | — | Run Exp 4 (GoEmotions GRPO). | ⏳ |
| S13 | — | Run `09` → consolidate into §9 parity schema + comparison table. | ⏳ |

✔ 2026-10-02 — **Installation:** `uv` already present; installed `prime` 0.9.1 + `hf` (huggingface-hub 2.1.1) via `uv tool install`. Confirmed auth commands: `prime login` / `prime config set-api-key`; HF: `hf auth login`.

✔ 2026-10-02 — **Compute recon (B2):** `prime pods`/`prime images`/`prime volumes` confirmed; `prime availability list` = live on-demand stock (A100 40GB ×4, A10 ×1, H100 80GB volatile). Full-FT still empty; volumes closed-beta. → **D16** (PI pod + self-run prime-rl).

✔ 2026-10-02 — **GoEmotions (B3):** Kaggle CLI absent + no token; fetched canonical `google-research-datasets/go_emotions` (parquet) instead, converted via `convert_goemotions_parquet_to_tsv.py`, built `goemotions/{chat,rl}/{train,dev}.jsonl` (800/200; 28 labels, ids 0–27 verified).

✔ 2026-10-02 — **SSH key diagnosis + fix (root cause found):** pod `0c22b573…` key injection was actually **working** — verbose ssh showed `Server accepts key: …SHA256:3evIu9mlfnw4saD3yNrMma+/sT1cQABOXVldfZBmpJY` (public key present in authorized_keys), but the local `~/.ssh/id_ed25519` is **passphrase-protected**, so non-interactive ssh couldn't sign (`we did not send a packet, disable method` → "Permission denied"). Fix: generated passphrase-less `~/.ssh/pi_id_ed25519`, registered as `pi-pod-key` (API auto-promoted it `isPrimary: true`, demoted `holbox-macbook`), `prime config set-ssh-key-path`, terminated `0c22b573…` and recreated `234fc42f…` (A100 40GB, us-east-1). No raw keys pasted.

✔ 2026-10-02 — **prime-rl on-pod discovery + config fixes (headless gotchas):** prime-rl lives at **`/app`** (not `/workspace/prime-rl`). The pod image ships an **older prime-rl** than GitHub main → `[run]` section is rejected (`Extra inputs are not permitted`); the run name lives in `output_dir` instead. `[data] name` is passed to `datasets.load_dataset(name, subset, split)` → a bare `.jsonl` **file** path fails (`Couldn't find any data file`); must point `name` at the **directory** containing `train.jsonl`. Configs rewritten accordingly (`output_dir=/workspace/outputs/…`, `name=/workspace/data/goemotions/chat`).

✔ 2026-10-02 — **Smoke SFT passed** (fake data, `Qwen/Qwen3-0.6B`, 10 steps, A100 40GB): config validated, model downloaded, LoRA+optimizer+scheduler+training loop all work; peak 11.1 GiB, ~936 tok/s. → **Exp 3 launched** (GoEmotions SFT, 100 steps, LoRA r=16, batch 64/µ8, cosine warmup 10). Observed learning: loss 5.79 → 3.37 over steps 1–6; `Qwen3Renderer` auto-selected; LoRA = 10.1M/440M params; peak 3.3 GiB.

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

---

## Results (empty until we have them)

| Exp | Metric | Fireworks (ref) | Prime Intellect | Δ |
|---|---|---|---|---|
| 1 ORena SFT | meanAccuracy (4-bucket) | — | — | — |
| 2 ORena RL | meanAccuracy (4-bucket) | — | — | — |
| 3 GoEmotions SFT | micro-F1 | — | — | — |
| 4 GoEmotions RL | micro-F1 / Jaccard | — | — | — |

---

## Append convention

When a decision is made or a step completes, update this file in the same commit/turn:
- flip a `⏳`/`🚫` to `✔`,
- add a dated row to the step log,
- add the number to the results table when a score lands.

Do not delete resolved rows — the point is the full trail from now to results.
