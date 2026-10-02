# Canonical input data

This folder holds the datasets that all three platforms run on. The files must
be byte-identical on every platform. Never re-split, re-shuffle, or re-sort a
file when you move between platforms.

Every dataset file is listed in [`manifest.json`](manifest.json) with its
`sha256`, row count, byte size, and source.

## GoEmotions (text classification)

- `data/train.tsv` (43,410) / `dev.tsv` (5,426) / `test.tsv` (5,427)
- Row format: `text \t comma-separated-label-ids \t id`
- 28 labels (27 emotions + neutral), listed in `data/emotions.txt`
- Multi-label: each example has 0..N labels
- License: Apache-2.0 (Google Research GoEmotions)

Derived files (also committed):

- `goemotions_sft.jsonl` — OpenAI CHAT JSONL, built from `train.tsv` + `emotions.txt`
- `goemotions_rl.jsonl` — RL JSONL, 5,000-row subset with `ground_truth.labels`

## ORena (surgical VQA — vision + text)

- Rows carry a base64 JPEG image inline, so the full JSONL files are large
  (up to 362 MB). They are **not** committed.
- `data/orena/orena_sft_sample_2rows.jsonl` and
  `data/orena/orena_rl_sample_2rows.jsonl` show the exact JSON structure.
- Regenerate the full files with `fireworks/scripts/20_build_datasets.py`, then
  check the `sha256` against `manifest.json`.

## Regenerating the large ORena files

```bash
# From the project that owns the source data:
python fireworks/scripts/20_build_datasets.py

# Verify (must match manifest.json exactly):
sha256sum data/orena/orena_sft.jsonl data/orena/orena_rl.jsonl
```

If the hashes do not match, do not run — the comparison against the other
platforms is only valid when the inputs are identical.
