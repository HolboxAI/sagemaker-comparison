# Analysis

Shared code and the canonical metrics schema. All comparison of the three
platforms happens here.

- [`schema.md`](schema.md) — the exact JSON schema every platform writes its
  `final_results.json` in.
- [`plot_results.py`](plot_results.py) — reads a platform's metrics and writes
  charts to `plots/`. Run it after each platform adds data.
- [`plots/`](plots/) — generated PNG charts (Fireworks results so far).
- `compare.py` (to add) — joins the three platforms' `final_results.json` files
  and prints side-by-side metric and cost tables.

## How comparison works

Each platform writes:

```
<platform>/metrics/final_results.json
```

A single script reads all three and joins them on the `experiment` key. That is
why the schema must match exactly — the join is on identical field names.

## Fields to compare

For every experiment, side by side:

1. **Performance** — final metric (train loss, exact_match, micro_f1).
2. **Cost** — `estimated_cost_usd` (and per-token / per-GPU-hour where the
   platform exposes it).
3. **Time** — `wall_time_s`.
4. **Feature gaps** — anything the platform could not do, recorded as a finding.
