# Baseten — experience report (CholecT50 SFT → RLVR)

**Date:** 2026-10-12
**Status:** blocked at authorization. The full pipeline was built and is push-ready,
but no training job ever started.

## 1. The task

Train a vision-language model to recognize **surgical action triplets**
(`instrument-verb-target`) in CholecT50 endoscopic frames, in two stages:

1. **SFT** — supervised fine-tuning to teach the output format.
2. **RLVR** — GRPO with a custom reward that scores how close the predicted
   triplets are to ground truth (dense triplet-F1). The demo goal is a measurable
   before/after F1 improvement.

Model: `Qwen/Qwen3-VL-4B-Instruct` (small, ungated, non-thinking — chosen so the
reward is well-behaved).

## 2. What we built (all push-ready)

- `config/config_sft.py`, `config/config_rl.py` — `truss_train` launch configs.
- `scripts/sft.py`, `scripts/grpo.py` — training entry points.
- `scripts/reward.py` — the triplet-F1 reward (unit-tested).
- `scripts/cholec_data.py` — fetches only the manifest-listed frames from the 62.7 GB
  remote zip via HTTP range requests (~1.9 GB total), builds trl SFT/GRPO datasets.
- `scripts/eval.py` — greedy-decode + F1 before/after eval.

## 3. Problems we hit

| # | Problem | What we tried | Result |
|---|---|---|---|
| 1 | `baseten train push` → HTTP 403 "You are not authorized for Baseten training" | re-auth + re-push | same 403 |
| 2 | Assumed it needed a payment method | user added one, re-pushed | same 403 |
| 3 | Assumed it needed an HF token (model gating) | added `hf_access_token` secret | same 403 |
| 4 | Root-caused via the raw API | checked `/v1/users/me` | no training role/entitlement on the account |

## 4. Root cause

Training Jobs is an **org-level entitlement** on Baseten. The account's org does not
have it enabled, so the API rejects every training project creation with 403 —
before any compute is allocated or billed. The error is a bare 403 with no message
about *why*, which makes it look like a token/config problem when it isn't.

## 5. What this means

- **No compute was billed** (the 403 happens before provisioning).
- The code is **complete and push-ready**; the only open item is an account-side
  entitlement, not an engineering one.
- Cost if it had run: a single H100 at $6.50/hr. A small 1-epoch SFT plus a
  150-step GRPO run is likely **a few tens of dollars** in total, but this was
  never reached.

## 6. Recommendation

Ask Baseten support to enable **Training Jobs** for the org. Then:

```bash
baseten train push --config ./config/config_sft.py
baseten train push --config ./config/config_rl.py
```

No code changes needed.
