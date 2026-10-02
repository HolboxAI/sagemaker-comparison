# ORena-FOCUS RL finetune — Fireworks RFT port

The same job as `environments/orena-focus-rl-v2` (Prime Hosted Training), ported
to Fireworks Reinforcement Fine-Tuning. Same task, same filters, same scorer —
`orena_scoring.py` is byte-identical across both projects (sha256 `bf9ea4f3…`,
asserted in `test_reward.py`), so reward numbers are comparable platform to
platform.

## Why this is a port and not a config swap

Prime runs a live `verifiers` environment: prompts are built inside the rollout
worker, so it can stream a private HF dataset and downscale frames at rollout
time. Fireworks RFT takes a **static JSONL dataset** plus an **evaluator
function**. That moves three things offline:

| | Prime | here |
|---|---|---|
| prompt construction | in the rollout worker | `export_dataset.py`, offline |
| images | fetched + resized per rollout | base64 JPEG baked into the JSONL |
| `HF_TOKEN` | needed on the platform | only needed on your laptop |
| reward | `vf.Rubric(funcs, weights)` | `@evaluation_test` → `EvaluateResult` |
| model | Qwen3.5-9B | `qwen3-vl-8b-instruct` |

The token change is a genuine improvement: on Prime a missing/expired `HF_TOKEN`
killed every rollout at step 0. Here the dataset is already materialized, so
that failure mode does not exist.

The cost is that **`image_max_side` is baked in**. Changing it means re-exporting
and re-uploading, not editing a config.

## Files

    export_dataset.py   HF dataset -> Fireworks RFT JSONL. Applies the filters.
    evaluation.py       the evaluator (Eval Protocol). Mirrors the Prime rubric.
    test_reward.py      42 offline tests. No GPU, no API key, no network.
    orena_scoring.py    vendored verbatim from the Prime env. Do not edit here.
    requirements.txt    evaluator deps only — one line. `ep upload` hard-fails
                        without this file even when it is empty.

## Model: one option, and it is not the Prime one

Vision RFT on Fireworks supports exactly one training shape —
**`qwen3-vl-8b-instruct`** (`trainingShapes/qwen3-vl-8b-65k`). There is no
choice to make, but two things follow from it:

* It is a different family from the Qwen3.5-9B that wedged after the first
  weight update on all eight Prime runs. That is arguably the point of trying
  another platform: the one variable those runs never isolated was the model
  family itself.
* **No VL model is serverless on this account.** Every vision model in the
  catalog returns `404 … not deployed` from the inference endpoint. Managed RFT
  provisions its own rollout capacity so this should not block training, but it
  does mean you cannot cheaply smoke-test rollouts against a live VL endpoint
  first. That is why reward correctness is covered offline instead.

RFT is documented as free under 16B parameters, which an 8B model satisfies.
Confirm on the dashboard before launching — do not take this file's word for it.

## Step 0 — verify the reward before spending anything

A subtly wrong reward still produces a beautiful training curve. Run this first;
it needs nothing but Python:

    python -m pytest test_reward.py -q      # 42 passed

The load-bearing assertion is that a gold answer scores **1.0 against itself**
for every answer format. If that fails the task is unlearnable and every dollar
after it is wasted.

Already verified against 8 real validation rows end to end: gold-as-prediction
scored `[1.0]*8`, garbage scored `[0.0]*8`, and the base64 frames decode to
512-px JPEGs inside a parsed `EvaluationRow`.

## Step 1 — export

    python export_dataset.py --split train --max-rows 2000 --seed 42 \
        --out data/orena_train.jsonl

    python export_dataset.py --split validation --max-rows 200 --seed 42 \
        --no-filters --out data/orena_eval_unfiltered.jsonl

For the **full** training set (all 10,980 rows surviving the filters), drop the
cap — this is what run 3 onward uses:

    python export_dataset.py --split train --max-rows -1 --seed 42 \
        --out data/orena_train_full.jsonl        # 10,980 rows, 366 MB

    format mix: FO_Class 5659 | number 4116 | binary 1205

Chunk size is 200, so row count sets curve granularity: 2,000 rows gave 10
datapoints, 10,980 gives ~55.

Measured: **~33 KB/row**, so 2000 rows ≈ 63 MB. Budget roughly 32 MB per 1000
rows at `--image-max-side 512`; halving the side quarters the vision tokens and
roughly quarters the file.

`--seed 42` matters. Unshuffled, the dataset runs in video order, so a batch
would be consecutive frames of one procedure — near-duplicate rollouts, and
group-relative advantage collapses when every member of a group is the same
frame.

`--no-filters` on the eval set is deliberate: it is the unfiltered base
distribution across all five answer formats, which is what the ORena test set
resembles. **Report that number, never the training-set number.** Expect a gap.
A gap that *widens* over training means the filters bought in-distribution
reward at the cost of the formats that were removed.

Filter counts, for reference:

    train      13889 -> 12279 (format) -> 10980 (class) -> 2000 (cap)
    validation  3298 ->  2873 (format) ->  2407 (class)

Training mix as exported: FO_Class 1041 / number 732 / binary 227.

## Step 2 — upload, register, launch

**These commands must be run by you, in your own terminal.** Fireworks blocks
mutating `firectl` calls made from inside an AI agent:

    BLOCKED: mutating command "firectl create dataset" cannot run inside an AI agent

Set `FIREWORKS_API_KEY` first; every command below reads it from the environment.
Do **not** pass `--api-key` on the command line — firectl echoes it in error
messages.

    # 1. datasets
    firectl create dataset orena-focus-train-2000 data/orena_train.jsonl
    firectl create dataset orena-focus-eval-200   data/orena_eval_unfiltered.jsonl

    # 2. evaluator (from this directory)
    .venv/bin/ep upload --entry evaluation.py::orena_focus_reward --yes \
        --display-name "ORena-FOCUS foreign-object VQA"

    # 3. the job — add --dry-run to see the request without sending it
    .venv/bin/ep create rft --yes --skip-validation \
        --training-config-base-model accounts/fireworks/models/qwen3-vl-8b-instruct \
        --dataset orena-focus-train-2000 \
        --evaluation-dataset accounts/om-chikhaliya-j5wmcp/datasets/orena-focus-eval-200 \
        --evaluator accounts/om-chikhaliya-j5wmcp/evaluators/evaluation-orena-focus-reward \
        --training-config-output-model accounts/om-chikhaliya-j5wmcp/models/orena-focus-rl-v1 \
        --training-config-epochs 1 \
        --training-config-lora-rank 16 \
        --training-config-learning-rate 1e-4 \
        --inference-parameters-response-candidates-count 16 \
        --inference-parameters-max-output-tokens 512 \
        --inference-parameters-temperature 0.7 \
        --inference-parameters-extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'

**Fully qualify every resource name.** `--dataset` is auto-expanded to
`accounts/<acct>/datasets/<id>`, but `--evaluation-dataset` and
`--training-config-output-model` are not, and a bare name fails server-side with

    error parsing output model name: ... must be in the format
    "accounts/<accounts-id>/models/<models-id>"

**`--skip-validation` is mandatory here, not optional.** Without it the CLI runs
a local pre-flight that issues *real rollouts* through litellm — and no VL model
is serverless on this account, so every one 404s with `not deployed`. What that
pre-flight would prove is "the model can do rollouts", which is exactly the
thing that cannot be tested locally without paying for a dedicated deployment.
Reward correctness is covered by `test_reward.py` instead.

**`--yes` uploads secrets before `--dry-run` takes effect.** It will push
`FIREWORKS_API_KEY` from your environment into the account's secret store even
on a dry run — `--dry-run` only guards the final job-creation call. If you rotate
the key, update or delete that secret too, or the evaluator keeps being handed a
dead one. Drop `--yes` to get an interactive prompt you can decline.

**`--loss-config-method` is broken.** It rejects `GRPO` even though the flag's
own type is `Literal['METHOD_UNSPECIFIED','GRPO','DAPO','DPO','ORPO',
'GSPO_TOKEN']`, and rejects the lowercase `grpo` that firectl's help documents.
Omit it — GRPO is the default.

### Where each setting came from

`response-candidates-count 16` is `rollouts_per_example = 16` from the Prime
config: large groups because the reward is dense and near-miss counts need
spread for group-relative advantage to have anything to work with.

`max-output-tokens 512` and **thinking off** were measured on Prime, not
guessed. Two runs were burned discovering this:

    thinking on  @1024   69% of step-0 rollouts truncated
    thinking on  @2048   49% truncated at step 0, 44% at step 1
    thinking on  @3072   21% truncated (direct probe)
    thinking OFF         0/24 truncated, median ~2 output tokens

Head-to-head on 24 real validation frames: OFF scored correctness 0.286 /
exact 0.250, ON@3072 scored 0.223 / 0.167. n=24 is far too small to claim OFF is
more *accurate* — 6 vs 4 correct sits inside noise. The decisive part is that it
is not detectably worse at ~900× fewer output tokens. The answers are one
integer, one word, or a short class list; the reasoning was the entire cost
driver. **Do not turn thinking back on without re-running that probe.**

`lora-rank 16` is a step up from the default 8, on the theory that a vision task
needs more capacity than a text one. It is the least evidence-backed number
here; 8 is a reasonable thing to try if the curve is noisy.

## Results — run `jzsltk8o`, 2026-07-30

`qwen3-vl-8b-instruct` → `accounts/om-chikhaliya-j5wmcp/models/orena-focus-rl-v2`.
One epoch, 2000 filtered examples, 16 candidates, LoRA rank 16, lr 1e-4,
max_output_tokens 1024. Ten chunks, **zero rollout errors and zero length
truncations** across ~32,000 rollouts.

    Score              0.2246 0.2498 0.2654 0.3180 0.2835 0.3026 0.3155 0.3279 0.3078 0.3572
    exact_match        0.1925 0.2125 0.2266 0.2787 0.2553 0.2753 0.2853 0.2903 0.2619 0.3200
    no_excluded_class  0.9803 0.9875 0.9812 0.9888 0.9900 0.9872 0.9941 0.9900 0.9928 0.9922
    no_hallucinated    1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000

`exact_match` 0.1925 → 0.3200 (+0.128, +66% relative). Each chunk is 200 unique
examples, so the 1-SE sampling floor is about ±0.031 — and that is optimistic,
since 16 rollouts per example are correlated. The two mid-run pullbacks (chunk 4,
chunk 8) are both under 1 SE and mean nothing individually; the end-to-end gain
is roughly 4 SE and does.

The 0.1925 starting point independently corroborates the Prime measurement: the
unbiased step-0 band there was 0.17–0.25. Two platforms, two model families,
same baseline — evidence the port is faithful rather than merely functional.

`no_excluded_class` rose 0.9803 → 0.9922. Small and near ceiling, but rising:
the silicone-loop prior weakened under filtering instead of surviving in the base
weights. That was the question the filtering existed to answer, and no Prime run
survived long enough to produce a second datapoint for it.

### 0.3200 is NOT the number to report

`output_stats` for "EVAL epoch 0" carries `n=3200` — 200 examples × 16
candidates — and an `exact_match` identical to the final training chunk to four
decimals. It is the last training chunk, not a held-out evaluation.
`--evaluation-dataset` did not produce an independent figure.

So 0.3200 is measured on the **filtered** distribution, which excludes
`open_ended` and `multiple choice` — the two hardest formats. The honest ORena
number is lower and still unmeasured. Getting it requires deploying
`orena-focus-rl-v2` and running the evaluator against `orena-focus-eval-200`,
which costs an on-demand GPU because no VL model is serverless on this account.

Quoting 0.3200 as ORena performance would repeat the selection-bias error made
on Prime, where a step-0 figure of 0.550 turned out to be conditioned on
rollouts that completed — which preselects easy frames.

### Why this ran when Prime did not

Eight Prime runs wedged after the first weight update: steps 0–1 clean on base
weights, then a ModelError flood with the run still `RUNNING` and still billing.
Environment, modality, max_tokens, thinking, batch geometry, checkpoint restart
and model size were all ruled out. The one variable those tests could not vary
was the model family, because Prime offered no vision alternative to Qwen3.5.

This run cleared ten consecutive chunks past that wall on Qwen3-VL-8B. That is
consistent with the trigger being Qwen3.5-on-Prime, though a single run on a
different platform *and* a different family cannot isolate which of the two
mattered.

## The reward

Only `correctness` moves the gradient. The rest are zero-weight monitors that
ride along in `EvaluateResult.metrics`, exactly as on Prime:

| weight | metric | |
|---|---|---|
| 1.0 | `correctness` | dense, format-aware. The training signal. |
| 0.0 | `exact_match` | strict 0/1. **The number you report.** |
| 0.0 | `no_excluded_class` | still naming silicone loop after filtering? |
| 0.0 | `no_hallucinated_class` | see below — this one is a no-op. |

`no_hallucinated_class` **can never fire.** `parse_class_set` only ever emits
canonical vocabulary entries, so `named - known` is always empty and the metric
is pinned at 1.0. It was a no-op on Prime too. It is kept for ledger-shape
parity and asserted as a no-op in `test_reward.py` rather than quietly deleted —
but do not read it as evidence the model isn't inventing classes.

`no_excluded_class` is the one worth watching. Holding at 1.0 while
`exact_match` climbs means the export filter worked. Decaying means the silicone
loop prior survives in the base weights and filtering alone will not remove it.

## Gotchas

**No `from __future__ import annotations` in `evaluation.py`.** Eval Protocol
validates the decorated function by comparing `row`'s annotation against the
`EvaluationRow` class; PEP 563 turns it into the string `"EvaluationRow"` and
registration fails with `In pointwise mode, the 'row' parameter must be of type
EvaluationRow` — which points at the wrong thing entirely.

**Images must be base64 with a MIME prefix.** Fireworks rejects plain `http(s)`
URLs in training data, and accepts PNG and JPEG only.

**`ep upload` tars this entire directory** and ships it to the evaluator build,
honouring `.gitignore` on top of its own defaults (which already cover `.venv`).
Two consequences: `orena_scoring.py` travels with the evaluator, which is why
the remote `import orena_scoring` works — and anything you drop in here gets
uploaded unless it is ignored. Keep `data/*.jsonl` and `.venv/` in `.gitignore`;
without them the payload goes from 48 KB to roughly 470 MB.

The remote build installs `eval_protocol` but **not** `datasets` or `pillow`,
so `export_dataset.py` imports those lazily inside its functions. Module-level
imports there would break collection of a file the remote never runs.

**Filters live in two files.** `TRAIN_FORMATS` / `EXCLUDE_CLASSES` in
`export_dataset.py` and `EXCLUDED_CLASSES` in `evaluation.py`. They are asserted
equal by `test_excluded_set_matches_the_exporter`; if they drift, the monitor
reports on a different set than was actually removed.

**`data/*.jsonl` is gitignored.** This repo is rooted at `$HOME`, and the
training set is 63 MB of base64.
