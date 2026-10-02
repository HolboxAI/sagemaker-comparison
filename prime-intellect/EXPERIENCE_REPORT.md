# Prime Intellect — Headless SFT/RL Experience Report

**Date:** 2026-10-02
**Scope:** Reproduce 4 experiments (ORena SFT/GRPO, GoEmotions SFT/GRPO) from a Fireworks reference config on **Prime Intellect**, driven entirely **headless** (CLI + REST API, no dashboard).
**Audience note:** This documents the *headless* path (`prime` CLI v0.9.1 + `api.primeintellect.ai`). The hosted/web platform may behave differently than what we hit below.

---

## 1. What we were asked to do (prompts, in order)

| # | Prompt | Intent |
|---|---|---|
| 1 | Reproduce the 4 experiments in `FIREWORKS_4EXP_CONFIG.md` on Prime Intellect (not Fireworks); small runs (≤20–30 min each); everything in `prime_intellect/`; one doc per step; capture screenshot + loss/reward plot + final result; tensorboard/CSV, no W&B | Overall brief + evidence + logging constraints |
| 2 | "run it" | Drive pod creation + training to completion |
| 3 | "please explain the error" | Explain SSH `Permission denied` |
| 4 | "i have only prime intellect api key… what should i do?" | Auth provisioning (API key only, no OAuth) |
| 5 | "register the key" | Explicit authorization to register the SSH public key via API |
| 6 | "update?" / "check again" (×several) | Progress checks during pod provisioning |
| 7 | "make a report from this chat…" | This document |

The compute and data-source decisions were made interactively before the run phase (see §3).

---

## 2. Queries / commands used (headless surface)

**CLI (`prime` 0.9.1):**
- `prime config view`, `prime config set-api-key`, `prime config set-ssh-key-path`
- `prime availability list --plain` — live on-demand GPU stock
- `prime pods create --cloud-id … / --id … --image prime_rl --name … -y --plain`
- `prime pods status <id>`, `prime pods ssh <id>`, `prime pods terminate <id> -y`
- `prime pods create --help` / `prime pods ssh --help`

**REST API (Bearer auth from `~/.prime/config.json`):**
- `GET  /api/v1/ssh_keys/`
- `POST /api/v1/ssh_keys/`  (`{"name","publicKey"}`)
- `GET  /api/v1/pods/`

**On-pod (the actual training surface):**
- `uv run sft @ /workspace/configs/smoke_sft.toml` (entrypoint lives in the `prime_rl` image at `/app`, not `/workspace`)

---

## 3. Decisions locked (from `DECISIONS_LOG.md`)

- **Compute:** no own GPU box → **PI on-demand pod + self-run `prime-rl`** (full fine-tune closed/gated; hosted LoRA SFT gated on closed-beta volumes). Workhorse = **A100 40GB** ($1.99/hr); H100 80GB appears but sells within ~2 min.
- **Data:** GoEmotions fetched from canonical HF `google-research-datasets/go_emotions` (Kaggle source needs a token). 800 train / 200 dev rows.
- **Models:** `Qwen/Qwen3-0.6B` (text, GoEmotions), `Qwen/Qwen3.5-9B` (VLM, ORena).
- **Secrets:** never paste raw keys into chat; set via masked CLI.
- **Run order:** de-risk with GoEmotions SFT (Exp 3) first.

---

## 4. Failures, root causes, and retries

This is the core of the "ease of use" story.

### 4.1 Pod creation — volatile inventory (4 attempts)
| Attempt | Command | Result |
|---|---|---|
| 1 | `--gpu-type "A100 40GB" --gpu-count 1` | "No configuration found for 1x A100 40GB" |
| 2 | `--id b203f6` (short hash from list) | "No valid GPU configuration found" (offer sold) |
| 3 | `--cloud-id gpu_1x_a100_sxm4` | worked → pod `0c22b573…` |
| 4 | `--cloud-id gpu_1x_a100_sxm4` (again, ~30 min later) | "No valid GPU configuration found" (went stale) |
| 5 | `--id e2c2b8` (fresh from `availability list`) | worked → pod `234fc42f…` |

**Root cause:** on-demand offers are ephemeral. Both the short `--id` and the "stable" `--cloud-id` can go stale in minutes. A headless driver must re-query `availability list` immediately before `create` and retry.

### 4.2 SSH auth — the big one (2 pod recreations, 1 key re-registration)
- **Symptom:** `Permission denied (publickey,password)` on every `ssh` / `prime pods ssh`.
- **Wrong hypothesis #1:** "no key registered" (because we authenticated with `prime config set-api-key`, not OAuth `prime login`). We registered `~/.ssh/id_ed25519.pub` via `POST /ssh_keys/` (HTTP 201) and recreated the pod. **Still failed.**
- **Actual root cause (found with `ssh -vv`):** the pod **had** our public key (`Server accepts key: … SHA256:3evIu9mlfnw4saD3yNrMma+/sT1cQABOXVldfZBmpJY`), but our **local private key is passphrase-protected**, so non-interactive SSH could not sign (`we did not send a packet, disable method`). Key injection had worked all along.
- **Fix:** generate a dedicated passphrase-less `~/.ssh/pi_id_ed25519`, register it (auto-promoted `isPrimary`), `prime config set-ssh-key-path`, terminate + recreate the pod.

**Headless lesson:** a passphrase-protected key silently breaks automation; the failure looks identical to "wrong key" unless you run `ssh -vv`. The platform's OAuth login path presumably avoids this — the API-key path does not.

### 4.3 prime-rl config schema drift (1 rewrite)
- We authored configs against GitHub `main` (which has `[run]`, `[renderer]`, etc.).
- The `prime_rl` pod image ships an **older** prime-rl: `[run]` was rejected with `Extra inputs are not permitted`.
- **Fix:** read the pod's *installed* schema (`/app/packages/prime-rl-configs/src/prime_rl/configs/sft.py`) and drop `[run]` in favor of `output_dir`.

**Headless lesson:** there is no version pin surfaced up front; you must validate against the installed package, not the public docs.

### 4.4 Local JSONL data loading (1 fix)
- `[data] name` is passed straight to `datasets.load_dataset(name, subset, split=…)`.
- A **single `.jsonl` file path** fails: `FileNotFoundError: Couldn't find any data file at …/train.jsonl`.
- **Fix:** point `name` at the **directory** containing `train.jsonl` (HuggingFace `datasets` auto-detects the split file).

### 4.5 Small surprises
- prime-rl is installed at **`/app`**, not `/workspace/prime-rl` (the documented expectation).
- The Jupyter port (8888) was **not reachable** from the public IP, so there was no web-console fallback during the SSH-outage window.
- `uv run` as `root` triggered a one-time venv re-sync (uninstalled/installed packages) — harmless here, but worth running as the image's `appuser` for cleanliness.

---

## 5. Retry tally

| Area | Retries | Outcome |
|---|---|---|
| Pod creation | 5 attempts | 2 pods ultimately created |
| SSH key | 1 register + 1 regen/re-register + 2 pod recreations | solved |
| Config validation | 1 rewrite (after reading installed schema) | solved |
| Data path | 1 fix (file → directory) | solved |
| **Total pod recreations** | **2** | — |

No training run failed on a *compute* problem; all failures were **plumbing** (inventory, SSH, config schema, data path).

---

## 6. Current status (as of writing)

- Pod `234fc42f…` (A100 40GB SXM4, `prime_rl` image) is up; SSH works.
- GoEmotions JSONL + configs transferred; `load_dataset` verified (800 train / 200 dev).
- **Smoke SFT (fake data) passed** — Step 1 loss 9.27, peak 11.1/39.5 GiB; model `Qwen/Qwen3-0.6B` loads and trains end-to-end.
- Next: real Exp 3 (GoEmotions SFT), then ORena SFT/GRPO + GoEmotions GRPO.

---

## 7. Headless ease-of-use assessment

**What works well**
- `prime pods` lifecycle is simple (`create/status/ssh/terminate`); provisioning + image install is fully automatic.
- The `prime_rl` image is self-contained (CUDA, `uv`, the repo, a ready `.venv`) — once you're in, `uv run sft @ config.toml` just runs.
- Model download is fast; a single A100 40GB comfortably trains a 0.6B LoRA.
- REST API for SSH keys is clean and idempotent (re-register auto-promotes primary).

**What is rough headless**
1. **Volatile inventory** — offers stale in minutes; no reservation; driver must poll-and-retry.
2. **SSH key UX** — API-key auth + passphrase-protected local keys = silent failures; only `ssh -vv` reveals the truth.
3. **No version pinning** — the pod image's prime-rl can be behind the public docs; configs must be checked against the installed package.
4. **No `--wait` on create** — you must poll `status` yourself for IP/install completion.
5. **No web-console fallback** — Jupyter port wasn't reachable, so when SSH was broken there was no backdoor.
6. **Local data requires directory conventions** — single-file JSONL isn't accepted; the `datasets` directory format is required.

**Bottom line:** headless SFT on PI is *workable but not polished*. It took ~5 pod-create attempts and 2 pod recreations to get to a running training step — almost all of it plumbing, none of it compute. A team that (a) pins a prime-rl version, (b) uses a passphrase-less dedicated key, and (c) wraps `create` in a poll-and-retry loop would smooth out most of the friction. RL (GRPO) is untested as of this writing and is the next thing to de-risk.

---

## 8. Product-surface confusion (why the UI looked empty)

Prime Intellect exposes **two distinct compute surfaces**, and they don't share a dashboard:

- **Hosted Training** (`prime train …` = "jobs") — the managed training product with a web dashboard, metrics, checkpoints. This is what the UI's "Jobs/Training" view shows. For our account it is **empty by design**: hosted full fine-tune returns an empty GPU list, and hosted SFT requires closed-beta volumes — so we never used it.
- **On-demand Pods** (`prime pods …` = raw VMs) — a separate, largely CLI-first feature. **This is where all four experiments actually ran.**

Consequences we hit headless:
- Running an experiment as a *pod* produces **nothing** in the "Jobs" view — a user watching the dashboard sees an empty account despite an active, billable GPU.
- Credits **are** tracked and visible only via `prime wallet` (balance + billing rows); the dashboard may not reflect pod spend in the same place.
- The API-key account here is `purchase@holbox.ai` with "Username: Not set" — if the web UI is logged in under a different OAuth identity, that would compound the "empty dashboard" symptom.

**Recommendation:** when billing or monitoring matters, treat `prime pods list` + `prime wallet` (CLI) as the source of truth for pod-based work, not the dashboard.

*Platform/dashboard behavior may differ — this report reflects the CLI + API (headless) path only.*
