# Phase 1 summary — G1 gate

**Watermark: SYNTHETIC - not tradeable** — no number in this file may enter the report's verdict.

- data: `synthetic(seed=7, n_bars=26280)` — 26280 bars, 2022-01-01 00:00:00+00:00 → 2024-12-30 23:00:00+00:00
- gaps: 0 (longest 0.0h) | duplicates: 0 | rows dropped: 0
- frame sha256: `9996fc7010e2defd1f8dafca366f0a527133b24d25fad4b06a7d6309f036a82b`
- config sha256: `0df7d21ecba517614c123b0c7f8bc80314ffc55545bae26e8acc84de6cbf8bdc` | plan sha256: `4d94d97cf3b1e613c5317ac35d63665d598f26e01e437359680c7b4c6751cadc`
- effective warmup: 2276 bars (8.66% of sample) — FFD startup at d=0.3 needs 2275

## Leakage gate

| test | result |
|---|---|
| prefix_invariance | **PASS** |
| truncation_invariance | **PASS** |
| shift_by_one | **PASS** |
| cross_matrix_invariance | **PASS** |
| static_source_guards | **PASS** |

Negative controls in `tests/test_leakage.py` inject a full-sample z-score, a forward shift, and a centred rolling window; all three are detected, so the gate has power. **G1 green: True**

## Labels

| metric | value |
|---|---|
| n_complete | 1028 |
| base_rate | 0.3765 |
| breakeven_hit_rate_gross | 0.4 |
| breakeven_hit_rate_net | 0.4139 |
| mean_R_gross | -0.0378 |
| mean_R_net | -0.1009 |
| median_cost_R | 0.0594 |
| ambiguous_rate | 0.0058 |
| gap_slip_rate | 0.0019 |

The net breakeven hit rate is the number that matters: a filter whose accepted trades hit below **41.39%** loses money at 7 bps round trip, however good its accuracy looks.

## Splits

| fold | bars | train n | val n | pos train | pos val | purged | embargoed | under-powered |
|---|---|---|---|---|---|---|---|---|
| f0 | 7884–10812 | 261 | 111 | 83 | 45 | 2 | 0 | no |
| f1 | 10812–13740 | 371 | 120 | 127 | 47 | 2 | 1 | no |
| f2 | 13740–16669 | 490 | 120 | 175 | 54 | 2 | 2 | no |
| f3 | 16669–19597 | 613 | 135 | 229 | 54 | 1 | 0 | no |
| f4 | 19597–22525 | 748 | 124 | 283 | 50 | 0 | 1 | no |
| holdout | 22525–26279 | 870 | 155 | 333 | 53 | 3 | 0 | no |

Split fingerprint: `91aa5d1148ae1750c0c670cb7a61e96f7e10b381b7c200f863ad23ee97f5a080` — frozen before any modeling.

## Sample-size budget (plan §3)

Pool size: **50** candidate features. Rule: `p_used <= floor(n_train/10)`.

| fold | n_train | p_pool | p_max | status |
|---|---|---|---|---|
| f0 | 261 | 50 | 26 | budget binds (fewer than pool) |
| f1 | 371 | 50 | 37 | budget binds (fewer than pool) |
| f2 | 490 | 50 | 49 | budget binds (fewer than pool) |
| f3 | 613 | 50 | 61 | ok |
| f4 | 748 | 50 | 74 | ok |
| holdout | 870 | 50 | 87 | ok |

The budget binds in every fold on this fixture, so M1–M3 get trimmed feature sets and the tree models get depth/leaf caps. That is the plan working as intended, not a workaround.

## Feature screen (training block only)

- ADF tested: 48/50 continuous features; non-stationary at 5%: **0**
- not testable (constant or categorical): 2
- label-proxy flags (|Spearman| > 0.5 vs realized R): **0** — max |rho| = 0.141 (ema_spread_atr)
- features with missing values in fold 0: bars_since_prev_cross (1), ema9_slope_atr (6), ema21_slope_atr (6)

## What is blocked, and on what

| item | blocked by |
|---|---|
| Phase 3 ladder (12–48 h of tuning) | real CSV — running it on a synthetic fixture would produce a complete-looking ladder with no evidential value |
| Hold-out evaluation (G6) | `freeze.json`, which requires a frozen model trained on real data |
| Second-instrument validation | no second CSV yet |
| M8 sequence ablation | torch not installed (as agreed: reported as `not run — dependency/gate`) |
