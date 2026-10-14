# Miles — RL Setup Config

_Compiled 2026-10-14. Miles is the fourth platform in this benchmark. This leg
covers the two RL experiments (GRPO). The SFT experiments stay on Fireworks,
SageMaker, and Prime Intellect. Miles can do SFT too, but the ask was RL._

Miles is an open-source RL framework (fork of `slime`). Rollout runs on SGLang.
Training runs on Megatron-LM or PyTorch FSDP. You self-host it on a GPU node. It
is the same product shape as Prime Intellect (self-run), not a managed service
like Fireworks.

Repo: `github.com/radixark/miles`. Docs: `miles.radixark.com`.

---

## 1. Experiment matrix (this leg)

| # | Experiment | Dataset | Method | Miles base model | HF id | Params |
|---|---|---|---|---|---|---|
| 2 | ORena VLM RL | ORena train | GRPO | `Qwen3-VL-8B-Instruct` | `Qwen/Qwen3-VL-8B-Instruct` | 8.7B (VLM) |
| 4 | GoEmotions RL | GoEmotions train | GRPO | `Qwen3-4B` | `Qwen/Qwen3-4B` | 4.4B (text) |

Rules (same as the other platforms):
- Fresh runs. No warm-start.
- RL method = GRPO. `n_samples_per_prompt = 8`.
- Short runs. GoEmotions first (de-risk), ORena second.
- Same data, same reward semantics as Fireworks and SageMaker.

---

## 2. Reward functions (Miles custom RM)

Miles wires a reward through `--custom-rm-path <module>.<fn>`. The function has
the contract:

```python
def reward_fn(args, sample, **kwargs) -> dict:
    pred = sample.response      # final completion, str
    gold = sample.label         # ground truth from --label-key, str
    return {"reward_value": float, "reward_cat": str, ...}
```

`--reward-key reward_value` pulls the float out of the dict.
`--log-reward-category reward_cat` logs the category.

This is simpler than Fireworks. Fireworks wants a tarball upload plus Eval
Protocol decorators. Miles wants one Python file and one function. SageMaker TRL
wants a plain `reward_func(completions, ...)` returning a list. Miles is the
middle: one function, returns a dict, one float key.

The two reward files are ports of the Fireworks reference, byte-for-byte on
scoring logic:

| File | Objective | Reports |
|---|---|---|
| `miles/scripts/reward_orena.py` | dense format-aware correctness | `exact_match`, hallucination flags |
| `miles/scripts/reward_goemotions.py` | Jaccard of label set | `micro_f1` |

---

## 3. Data

Miles loads prompts from a JSONL via `--prompt-data <file> --input-key <k>
--label-key <k> --apply-chat-template`.

```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."}],
 "label": "{\"answer\": \"clip\", \"answer_format\": \"fo_class\", \"choices\": []}"}
```

`messages` is the full chat list (system + user). `label` is the ground truth
as a JSON string, because Miles `Sample.label` is a `str` (one key only). The
reward function parses that string back into a dict or list.

`miles/scripts/prepare_data.py` builds both files from the canonical `data/`
sources:

- `miles/data/goemotions_rl_miles.jsonl` — 5,000 rows, committed, runnable now.
- `miles/data/orena_rl_miles_sample.jsonl` — 2 rows, smoke only. The full
  `data/orena/orena_rl.jsonl` (2,000 rows, 66 MB of base64 images) is not
  committed. Regenerate it, then re-run prepare_data.

---

## 4. Launch

GoEmotions (de-risk): `miles/scripts/run_goemotions_rl.sh`
ORena (follow-on): `miles/scripts/run_orena_rl.sh`

Both run `python3 train.py` inside the Miles Docker image (`radixark/miles:latest`).
GoEmotions uses the FSDP backend (reads HF weights, no Megatron conversion).
ORena uses the Megatron backend plus a VLM generate function (like the
`examples/geo3k_vlm` recipe).

---

## 5. Findings that change the comparison

1. **VLM RL base model is an exact match.** Miles supports
   `Qwen3-VL-8B-Instruct`, the same model Fireworks used for ORena RL. This is
   the only platform pair in the benchmark with the same RL base model on both
   sides. SageMaker could not do this (it RL'd `Qwen3.5-9B` directly and hit the
   thinking-mode bug). Prime Intellect never reached RL.

2. **Text RL base model differs.** `Qwen3-0.6B` is not in Miles' supported model
   list. GoEmotions RL must use `Qwen3-4B`. Record this as a Miles catalog gap,
   the same class of finding as Fireworks (could not RL `qwen3p5-9b`) and
   SageMaker (GPU-class mismatch).

3. **Compute is self-host and heavy.** The standard Miles recipe wants one node
   with 8 H100/H200/B200 GPUs and Docker. The `riley` AWS profile has smaller
   GPU classes (L40S / L4 / Blackwell L-series). This is a GPU-class mismatch
   against Fireworks (B200/H200 managed), same as the SageMaker finding.

4. **Non-binary rewards can leak spurious gradient.** The Miles VLM recipe
   documents this: when every sample in a GRPO group shares the same fractional
   reward, `reward - mean` is not exactly zero under fp32, which adds noise. The
   reference team switched to clean 0/1 rewards to fix it. Our ORena reward is
   dense (fractional) and our GoEmotions reward is Jaccard (fractional), so this
   caveat applies to both. Fireworks and SageMaker accepted the same fractional
   rewards without flagging it.

5. **No upload step, no tarball, no managed dashboard.** Miles is a repo you
   pull, a container you run, and a `train.py` you launch. The closest managed
   analog is the reward function registration (`--custom-rm-path`), which is a
   single file path, not a signed upload.

---

## 6. Metric schema

Same `final_results.json` schema as the other three platforms
(`analysis/schema.md`). The RL curves are:

- ORena: `score_curve` (reward_value mean), `exact_match_curve`.
- GoEmotions: `jaccard_curve` (reward_value mean), `micro_f1_curve`.
