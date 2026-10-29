# double-ema-crossover-strategy-with-ml-filter

Meta-labeling filter for an EMA 9/21 crossover strategy on 1h OHLCV bars.

## Status

| Phase | State |
|---|---|
| 0 — modeling plan | **approved** with clarifications A1–A9 → [`docs/PHASE0_MODELING_PLAN.md`](docs/PHASE0_MODELING_PLAN.md) |
| 1 — data, labels, splits, leakage CI | **complete, G1 green** (synthetic fixture) |
| 2 — feature engineering | built + audited (stationarity, label-proxy check) |
| 3 — model ladder | **not started** (waiting on the real CSV before the ladder is worth running) |
| 4–6 | not started |

The pipeline runs end-to-end on real data or on a synthetic fallback that is watermarked
`SYNTHETIC - not tradeable` in every artifact. No synthetic number may enter the report's verdict.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt && ./.venv/bin/pip install -e .

make test                                  # 39 tests incl. the leakage gate + negative controls
./.venv/bin/python -m dema.cli audit --smoke-test         # G1 gate + label/split audits
./.venv/bin/python -m dema.cli audit --data /path/to/ohlcv.csv
./.venv/bin/python -m dema.cli features --smoke-test      # Phase 2 feature matrices
```

Artifacts land in `artifacts/`: `label_audit.md`, `split_audit.md`, `leakage.md`,
`feature_manifest.{csv,json}`, `splits.csv`, `manifest.json`, `features/`.

## What is guaranteed by code, not by intention

- **Entry is the next bar's open** — asserted in `labels.triple_barrier_labels`.
- **ATR is measured on the cross bar** (known at signal time), not the entry bar.
- **No feature reads a future bar** — proven by prefix/truncation invariance under
  future-noise replacement (`src/dema/leakage.py`), not by inspection.
- **No time-shuffling splitters** — forbidden by an AST/grep guard in CI.
- **Realized R never enters a feature** — features come from a declared registry only;
  a test asserts no label column is registered.
- **Purge + embargo ≥ max_hold** at every train/validation boundary, asserted per fold.
- **Under-powered folds are flagged, never dropped** (plan A6).
