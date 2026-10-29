# Phase 0 — Modeling plan for approval

**Strategy:** EMA 9/21 crossover, 1h OHLCV bars, ML meta-labeling filter.
**Status:** DRAFT — awaiting PI approval. **No model, feature, or pipeline code has been written.**
**Branch:** `arena/d2fd9e09-double-ema-crossover-strategy`
**Plan hash (pre-registration):** recorded in `docs/plan.sha256` at commit time, before any result is produced. Any post-approval change to this document is logged in §14 (Change log) with a reason.

---

## 0. Scope, gates, and the two things I need from you

### 0.1 Gates (work stops at each until the gate condition holds)

| Gate | Condition to pass | Deliverable / evidence |
|---|---|---|
| **G0** | You approve this plan | This document, approved |
| **G1** | Leakage CI green **and** split audit printed | `tests/test_leakage.py`, `artifacts/split_audit.md` — nothing trains before this |
| **G2** | Label sanity: base rate, same-bar ambiguity rate, overlap/concurrency report, R distribution | `artifacts/label_audit.md` |
| **G3** | Full ladder M0–M8 run, tuned, all reported (no skips) | `artifacts/ladder.csv` |
| **G4** | Diagnostics run for M1/M4/M5/winner; null test complete | `reports/figures/*`, `artifacts/null_dist.csv` |
| **G5** | Model + calibration + threshold **frozen** (hashed, one-touch ledger opened) | `artifacts/freeze.json` |
| **G6** | Single hold-out evaluation, three-arm backtest, report, PR opened (not merged) | `reports/report.md`, PR |

The hold-out is evaluated **once**, after G5. If I have to touch it twice, that is a defect and I will report both touches.

### 0.2 What I need from you (blocking for Phase 5, not for Phase 1–4)

1. **Real CSV path** + instrument (e.g. BTCUSDT, XAUUSD, ES) + exchange/source + **timezone** of the `timestamp` column.
2. Whether a **second instrument CSV** exists (determines whether §11.6 "not validated on one series" applies).
3. Cost conventions: is `cost_bps=5` per side or round trip? Default in this plan = **round-trip total 5 bps + configurable `slippage_bps`**, applied on notional at entry and exit. Tell me if your venue convention differs.
4. Permission to `pip install torch` (CPU wheel) for M8. Without it, M8 is reported as `not run — dependency unavailable`, not silently dropped. (The sandbox has 2 vCPU / 3 GB RAM; torch CPU is feasible, GPU is not.)

Until the CSV arrives I can build and validate Phases 1–4 on synthetic data, but every such artifact is watermarked `SYNTHETIC — not tradeable` and **no synthetic number may appear in the report's verdict or conclusion**.

---

## 1. Phase 1 — data, signal, labels (integrity foundation)

### 1.1 Loader
- Schema assertion: exactly `timestamp, open, high, low, close, volume` (+ optional extras tolerated but unused).
- Reject/repair rules, all logged: duplicate timestamps, unsorted rows, NaN/zero/negative prices, `high < max(open,close)`, `low > min(open,close)`, `volume < 0`.
- **Bar-grid check:** median Δt = 1h, and report the gap census (count of gaps > 1.5h, longest gap). Gaps are *not* silently forward-filled for return features — forward-filling creates fake zero returns and fake low vol. Instead: compute features on the true bar index and mark features that span a gap with a `has_gap_24` flag; rows with NaN after warmup are dropped, not imputed.
- Timestamps normalized to UTC; the raw tz is recorded in the manifest.
- `--smoke-test` generates a GBM+vol-clustering synthetic series, labels the axis, the manifest, the CSV filename, the report header, and every figure title with `SYNTHETIC — not tradeable`.
- `sha256` of the input file (and of the sorted frame) recorded in `manifest.json`.

### 1.2 Signal
- `ema9 = close.ewm(span=9, adjust=False)`, `ema21 = close.ewm(span=21, adjust=False)`. `adjust=False` (recursive, causal); `adjust=True` is banned because it uses a full-sample normalization.
- Cross at bar *t*: `sign(ema9−ema21)` changes between *t−1* and *t*. Direction = sign at *t*.
- **Entry at next bar open** — enforced by an assertion in the label function: `entry_idx == cross_idx + 1`, `entry_price == open[entry_idx]`. The entry bar's own high/low are usable for barriers (they occur after entry); the cross bar's are not.
- `warmup = 200` bars before the first eligible cross (EMA/ATR/72-bar-window convergence); asserted, and the count of discarded warmup bars is printed.
- Crosses on consecutive bars (possible on flat/quiet data) are kept but flagged `is_consecutive_cross`; sensitivity reported with them dropped.

### 1.3 Labels — triple barrier, per cross
- `ATR = ATR(14)` (Wilder) **at the cross bar** — the level is known at signal time, not at entry. Stated explicitly because using `ATR[entry]` would leak the entry bar.
- `TP = entry + side · tp_atr · ATR`; `SL = entry − side · sl_atr · ATR`; time barrier at `cross_idx + max_hold`.
- `y = 1` iff TP is touched strictly before SL within the horizon.
- Resolution conventions (all conservative, all counted and reported):
  - **Same-bar TP and SL touch → SL wins** (intrabar order unknown). Report `ambiguous_rate`; it is a real, quantified cost of 1h bars.
  - **Gap-through:** if a bar *opens* beyond a barrier, fill at the open (worse than the barrier). `R` can be < −1. Reported as `gap_slip_rate`.
  - **Neither touched** → exit at the close of the last bar in the horizon; `R` is whatever that is, including negative.
- `R = side · (exit − entry) / (sl_atr · ATR)` **gross**, and `R_net = R − cost_in_R` (§5.2). Stored for backtest only; **`R`, `R_net`, exit reason, and any function of forward bars are hard-excluded from the feature matrix** by an allow-list in the feature assembler (features are keyed by name from an explicit registry — there is no "use everything" path).
- Defaults in `configs/config.yaml`: `tp_atr 1.5, sl_atr 1.0, max_hold 24`. Sensitivity sweep per §11.5.

**Label arithmetic worth stating up front (it drives every result):** with TP 1.5R / SL 1.0R the gross breakeven hit rate is `1/(1+1.5) = 40%`. With the cost model of §5.2 at `cost_bps=5` and `SL ≈ 1×ATR ≈ 1–1.5% of price` on 1h bars, `cost_in_R ≈ 0.07–0.10R`, so the **net breakeven hit rate is ≈ 44%**. A model that "predicts profit" at 42% hit rate is worthless even though it beats a coin flip. This is why the selection criterion is expectancy-aware (§6), not accuracy- or even ROC-aware.

### 1.4 Splits
- **Purge + embargo ≥ `max_hold`** at every boundary, in **bars**, not rows: for any train/test boundary at bar index *b*, drop training crosses whose label window `[cross+1, cross+max_hold]` intersects `[b, b+embargo]`. `embargo = max_hold` (24 bars) by default, configurable.
- **Walk-forward**: expanding window, 5 folds (≥ 4 required), validation = the block immediately after each training block, separated by the purge/embargo gap. Outer folds never reorder time.
- **Hold-out**: the most recent contiguous block containing **≥ 40 crosses and ≥ 15% of crosses**, untouched until G5. Exact boundary is written into `freeze.json` before any modeling, and the boundary choice is made *before* seeing any label statistic from that block.
- `artifacts/split_audit.md` prints, per fold: bar range, timestamps, `n_train`, `n_val`, `n_purged`, `n_embargoed`, `n_test`, plus a machine assertion of `max(label_end[train]) < min(label_start[val])`.

### 1.5 Leakage test in CI (G1 — blocks training)
`tests/test_leakage.py`, run in CI and locally before every training invocation:

1. **Future-scramble (strongest):** replace all bars after index *t* with random-walk noise of identical shape, recompute the whole feature matrix, and assert features at index ≤ *t* are **bit-identical** to the original. This catches any accidental full-sample statistic (e.g. `scaler.fit(X)`, `quantile()` without `rolling`, `ewm(adjust=True)`).
2. **Shift-by-1:** shift the close series by one bar, recompute features, assert **every** feature changes at the cross bars (a feature that doesn't move is either a leak-free constant — reported — or a lookahead artifact of non-causal construction).
3. **Index-guard:** assert `feature_max_source_bar[f] ≤ cross_idx` for every feature *f*, using an instrumented accessor that records which bars were read while computing each feature. This is the mechanical version of "no feature uses bars > cross index".
4. **Post-hoc leak check:** for each feature, Spearman correlation with the *realized* `R` of the same trade. Any |ρ| above a threshold (reported, flagged, not auto-deleted) goes into the report's leakage table with a verdict. This is how label proxies get caught (§9.3).
5. **Fit-scope test:** assert every transformer/scaler/selector is fitted inside a `Pipeline` whose fit call sees only training rows (checked by fitting on a slice and asserting `transform` of the held-out block is unaffected by shuffling that block).

G1 is green only when all five pass.

---

## 2. Phase 2 — feature engineering spec

**Design principle:** features are computed on the full bar series but *read* only up to `cross_idx`. Every feature is registered in `FEATURE_REGISTRY` with name, window, formula, tier, and rationale; the model receives `X = df[selected_names]` where `selected_names ⊆ registry` — never a positional slice of the frame.

### 2.1 Base block (Tier A) — 21 features

| # | Feature | Formula (bar index ≤ cross) | Rationale |
|---|---|---|---|
| 1–5 | `ret_1, ret_3, ret_6, ret_12, ret_24` | `log(close_t / close_{t−k})` | Short-horizon momentum/mean-reversion at the cross; the local impulse that produced the cross |
| 6–7 | `rv_24, rv_72` | `std(ret_1, k) · √k` | Realized vol level; scales everything else and conditions TP reachability |
| 8–9 | `eff_24, eff_72` | `|close_t − close_{t−k}| / Σ|Δclose|` | Path efficiency: directionality of the move into the cross (target of the deterministic arm) |
| 10 | `autocorr_1_24` | `corr(ret_1_t.., ret_1_{t−1}..)` over 24 | Persistence vs chop regime |
| 11 | `atr_ratio_14_72` | `ATR14 / ATR72` | Vol expansion/contraction at the cross |
| 12–14 | `sma_gap_72_atr, sma_slope_24, sma_slope_72` | `(close − SMA72)/ATR`, `SMA72` diff over k / ATR | Distance from and slope of the trend anchor; stops the model treating all crosses as alike |
| 15–16 | `volume_z_24, volume_z_72` | `(vol − mean_k)/std_k` | Participation at the cross |
| 17–20 | `hour_sin, hour_cos, dow_sin, dow_cos` | cyclical encoding | Session/liquidity seasonality; continuous encoding avoids fake ordinality of 23→0 |
| 21 | `has_gap_24` | any bar gap > 1.5h in last 24 | Data-quality confounder, keeps gap artifacts from being read as vol |

### 2.2 Stationarity-robust transforms

- **72-bar rolling z-score and percentile rank** of every continuous Tier-A/B feature.
- **Volatility-normalized returns** `ret_k / rolling_std_1h_k` for `k ∈ {1,3,6,12,24}`.
- **Transform policy (deliberate, and it is a modeling decision you should check):** these do **not** double the matrix.
  - Tree models (M4, M5, M7 base trees, M8): **raw features only** — trees are invariant to monotone transforms, so ranks/z-scores add cost and correlated duplicates with zero information.
  - Linear models (M1, M2, M3): **rank-transformed version of each continuous feature in place of the raw one**, plus the 5 vol-normalized returns. Rank beats z-score for heavy-tailed crypto features and bounds the influence of outliers; z-scores are still computed and audited for stationarity but rank is what enters the model.
  - Rationale: keeping `p` identical across families makes the ladder comparison (which is the point of the ladder) apples-to-apples. Inflating `p` for linear models only would confound model choice with feature count and blow the sample-size budget in §3.

### 2.3 Fractional differentiation (my proposed implementation detail)

- Fixed-width-window FFD (López de Prado): weights `w_k = −w_{k−1}·(d−k+1)/k`, truncated at `|w| < 1e-5`, applied to `log(close)`.
- Candidates `d ∈ {0.3, 0.4, 0.5}`; **`d` selected by ADF p-value on training folds only** (per outer fold, so the selection itself is inside the fold). ADF p-values for all three are reported per fold.
- Exactly **one** FFD feature enters the model (the selected `d`). The other two are reported in the ADF table and discarded — feeding three near-collinear FFDs of the same series would spend the sample-size budget on a 0.1 change in `d`.
- Note in the report: at `d < 1` the series retains memory but also retains some trend; ADF is the arbiter, and if no `d` reaches stationarity at p < 0.05 I use the most stationary one and say so.

### 2.4 Cross-specific dynamics — 11 features

`bars_since_prev_cross` (min 1, log1p'd), `ema_spread_atr = |EMA9−EMA21|/ATR`, `ema_spread_pctile_72`, `ema9_slope_atr`, `ema21_slope_atr`, `cross_side ∈ {−1,+1}`, `dist_from_ema21_atr = (close−EMA21)/ATR`, `atr_pctile_72`, `volume_z_at_cross`, `volume_trend_12 = log(mean_vol_6 / mean_vol_12)`, `signed_volume_z_24 = Σ sign(Δclose)·vol_z / 24`.

Rationale: these describe *the cross itself* — how decisive it was (spread, slopes), whether it is a fresh move or a whipsaw (`bars_since_prev_cross`), and whether volume confirms it. `cross_side` is a first-class feature because long and short legs of a 1h EMA crossover are empirically not symmetric (drift, funding, volatility asymmetry), and a filter that cannot condition on side is forced to average two different problems.

### 2.5 Regime-conditional context — 3 additional

`eff_pctile_72`, `rv_pctile_72`, `dist_from_sma72_atr`. These convert level features into **regime-relative** ones so the model generalizes across vol regimes rather than memorizing absolute vol of one year.

### 2.6 Interactions (explicit, 4)

`cross_side × ema_spread_atr`, `cross_side × eff_pctile`, `cross_side × vol_pctile`, `eff_pctile × vol_pctile`.

Rationale: trees can find these but need depth and data; giving them explicitly (a) helps the linear models, (b) makes the fitted interaction available for the SHAP narrative, (c) costs 4 columns. I keep them for every family so the comparison stays clean.

### 2.7 Distributional — 4

`skew_24, skew_72, kurt_24, kurt_72` of `ret_1`. Tail shape conditions whether a 1.5-ATR target is reachable in a burst or needs a grind.

### 2.8 My proposed additions (beyond your spec) — each is optional, all are Tier B, and **the default recommendation is to drop 2 of them** on sample-size grounds

| Proposed | Formula | Why | My recommendation |
|---|---|---|---|
| A1 `clv_cross` | `(close−low)/(high−low)` at cross bar | Close location = rejection/wick asymmetry at the decision bar; 1 column, well-known reversal prior | **Keep** (cheap, distinct) |
| A2 `consec_cross_24` | count of crosses in last 24 bars | Whipsaw density; directly targets the failure mode of EMA crossovers | **Keep** |
| A3 `range_z_24` | `(high−low)/ATR` z-scored | Bar-range anomaly = news/liquidity event | Keep only if budget allows |
| A4 `hours_since_volume_spike` | log1p(bars since `vol_z_24 > 2`) | Participation bursts decay; captures staleness of the cross | Keep only if budget allows |
| A5 `session_flag` (Asia/London/NY) | from UTC hour | Strong for FX/indices, weak for crypto | **Drop by default** — it duplicates `hour_sin/cos` unless you tell me the instrument is FX/equity-index |
| A6 `funding/basis proxy` | — | Genuinely informative for perp futures | **Not possible** — not in OHLCV; listed in "what would make this better" |

Full pool after additions: **≈ 49 named features** (§3.1 counts the exact number at build time).

### 2.9 Feature table in the report
Every feature ships with: formula, window, tier, one-line rationale, permutation-importance rank (mean ± std across folds), Spearman ρ with realized R, SFI (single-feature-importance) verdict, and kept/dropped status with the reason.

---

## 3. Sample-size realism and the feature budget (the binding constraint)

A 1h EMA 9/21 cross fires roughly **0.35–0.6 times per day** (≈ 130–220/yr). Expected `n_crosses`:

| History | n_crosses (est.) | Usable p at n/p ≥ 10 | Plan response |
|---|---|---|---|
| 1 yr | 130–220 | 13–22 | **cap complexity: M1–M3 primary**, M4/M5 with `max_depth ≤ 4` and Tier A only; M7/M8 **not run** (M8 blocked by its own n ≥ 500 gate), M7 reported as skipped with reason |
| 2–3 yr | 260–650 | 26–65 | Full M1–M5; M7 if per-fold OOF is stable; M8 only if n ≥ 500 |
| 5 yr+ | 650–1100 | 65–110 | Full ladder incl. M8 |

**Budget rule (pre-registered, not tuned on results):** for each outer fold, `p_used ≤ floor(n_train / 10)`. If the candidate pool exceeds that, I drop features by a **pre-declared, training-fold-only** order of operations:

1. Drop mechanically redundant columns (|Spearman| > 0.95 pairs — keep the one with lower mean rank; ties broken by shorter window).
2. Drop near-zero-variance features (`std < 1e-6` or > 80% identical values).
3. Drop features with **training-fold** permutation importance in the bottom 30% (§9.2 protocol applied inside the training block only) — never using validation data.
4. If still over budget: increase regularization (`C` down / `max_depth` down / `min_samples_leaf` up) and record the tradeoff.

**Tradeoff reporting:** the report prints, per fold, `n_train`, `p_candidate`, `p_used`, the dropped list with reason, and the resulting `n/p` ratio. If the ratio is < 10 anywhere, that fold's complexity is called out as under-powered rather than quietly run.

**If `n_crosses < 300`:** M1–M3 become the primary ladder; M4/M5 are still run (you asked for no skips) but reported as *under-powered* with the depth/leaf constraints above; M7 stacking is **not** run because stacking OOF predictions from 5 base models on < 300 samples produces a meta-learner trained on noise — it will be listed in the ladder table as `skipped (n<300, stacking OOF unreliable)`. M8 gated at n ≥ 500 per your spec. A "sample-size budget" paragraph goes in the report either way.

---

## 4. Model ladder and search spaces

All models tuned with **Optuna TPE**, `n_trials ≥ 100`, sampler/seed fixed, MedianPruner on the inner CV mean. No sklearn defaults anywhere — every reported model uses an explicitly tuned best config, and the search space is printed in the report.

**Inner CV (all models):** purged, embargoed **3-fold walk-forward** *inside each outer training block* — expanding, contiguous, purge = `max_hold`, embargo = `max_hold`, no shuffling. Cross-validation *is* time-series cross-validation here; `KFold` and `StratifiedKFold` appear nowhere in the codebase (asserted by a grep test in CI).

| ID | Model | Search space | Notes / expected cost |
|---|---|---|---|
| **M0** | Majority-class baseline | none (it is a baseline; "tuning" it would be theater) | Predicts the training-fold majority; PR-AUC → prevalence. Reported for scale, with its own row in the table. |
| **M1** | Logistic, L2 **and** L1, scaled, `class_weight='balanced'` | `C ∈ logspace(−3, 2, 12)`, `penalty ∈ {l1, l2}`, `solver ∈ {liblinear, saga}`, rank-vs-raw transform, `class_weight ∈ {balanced, None}` | 100 trials × 3 inner folds ≈ 2–4 min/outer fold |
| **M2** | Logistic + RFECV selection | M1 space + `step ∈ {1,2}`, `min_features_to_select ∈ {5,8,12,16}`, `cv=purged 3-fold` | Report selected set **per outer fold** + Jaccard stability across folds. RFECV is nested inside tuning → this is the most expensive linear model (~20–40 min/outer fold) |
| **M3** | Logistic + exponential time-decay weights | M1 space + `half_life ∈ {250, 500, 1000}` bars, `C ∈ logspace(−3,2)`; weight `w_i = 0.5^((T−t_i)/half_life)` | Calibrate **after** (calibration fit on unweighted data — weighting then calibrating is the standard order) |
| **M4** | ExtraTrees | `n_estimators 300–1000`, `max_depth 3–12`, `min_samples_leaf 5–50`, `max_features ∈ {sqrt, 1/3, all}`, `min_samples_split`, `criterion ∈ {gini, entropy}`, `class_weight ∈ {balanced, balanced_subsample, None}` | ~6–10 min/outer fold at n≈600 |
| **M5** | HistGradientBoosting | `learning_rate 0.01–0.1` (log), `max_depth 2–6`, `l2_regularization 0–10`, `max_leaf_nodes 7–63`, `min_samples_leaf 10–60`, `max_iter 100–600`, `class_weight ∈ {None, balanced}`, `early_stopping=True` on the **purged inner CV** validation fold | Early stopping must read the inner purge-aware split, never the outer validation |
| **M6** | Soft-vote ensemble | Top-3 by **validation** PR-AUC (selection stage, see §6.3 for the honesty caveat) + weights on the simplex in 0.05 steps | Weights tuned on validation **only after** the top-3 are fixed; this is a second-stage selection — I will report it with the extra-selection-cost caveat and a nested-check: weights re-tuned inside each outer fold as a sanity comparison |
| **M7** | Stacking, logistic meta-learner | Meta features = OOF P(y=1) from M1–M5 + `cross_side`, `ema_spread_atr`, `eff_pctile_72`; meta space `C ∈ logspace(−3,2)`, `penalty ∈ {l1,l2}`, include/exclude passthrough features | **Nesting caveat, stated up front:** base OOF is generated *inside* each outer training block via the inner purged folds, with base hyperparameters frozen at their tuned values — re-tuning 5 bases inside every inner fold is ~10⁴ extra fits and cannot be done honestly in the compute budget. Tuning, therefore, is over the **meta-learner only**. This is a documented deviation from fully nested stacking; it inflates M7's in-sample optimism slightly, and I will say so in the report. |
| **M8** | Sequence ablation: 1D-CNN or GRU on last 24 bars of `ret_1`, `vol_z_24` | **Only if n_crosses ≥ 500.** Tunable: `arch ∈ {CNN, GRU}`, `hidden ∈ {8,16}`, `layers ∈ {1,2}`, `dropout ∈ {0,0.2,0.4}`, `lr ∈ logspace(−4,−2)`, `weight_decay ∈ logspace(−5,−2)`, `epochs 20–100` with early stopping on inner purged CV, batch size, seed-averaging over 5 seeds | Ablation only — never the recommended primary, because 24×2 inputs at n≈500 is a sample-size mirage. Reported as ablation with its own PR-AUC and the caveat. 100 trials over a reduced space (~10–20 min/outer fold on 2 CPU). If torch is unavailable → `not run — dependency`. |

**Seed averaging:** every reported tree/sequence model is refit with **5 seeds** per fold and averaged, so the table is not a lucky-seed artifact; the seed std appears in the artifact, not the headline table.

---

## 5. Tuning protocol

1. **Objective = mean inner-CV PR-AUC** (average precision of `y=1`) across the 3 purged, embargoed inner folds; secondary logged objective = mean inner expectancy. Optuna's `MedianPruner` prunes trials after inner fold 1 if clearly dominated → the "≥ 100 trials" requirement is on *completed* trials; pruned trials are counted and reported separately.
2. **No tuning on validation. No tuning on the hold-out. No tuning after the threshold freeze.** After G5, the hyperparameter file is hashed and read-only; the code path that reads it refuses to run if the hash changes.
3. **Param stability across outer folds:** for each hyperparameter, report the value chosen per outer fold, the modal value, and a normalized dispersion (IQR / range of the search space). Unstable params (> 50% dispersion) are flagged, and the report says plainly that the tuning surface is flat there — which matters more for credibility than the "best" value.
4. **Search-space report:** the full grid is emitted as `artifacts/search_spaces.json` and a table in the report.
5. **Random search fallback:** only if Optuna cannot be installed; the run manifest records which was used. (Optuna is installable here.)
6. **Runtime honesty:** estimated 2–5 h wall-clock for the ladder at n ≈ 600, p ≈ 30 on 2 vCPU; 4–10 h at n ≈ 1500. If the budget forces a reduction, I reduce **trials**, not models, and log `trials_completed` / `trials_pruned` per model in the report. No model is silently dropped.

---

## 6. Selection criterion — decided on validation only

### 6.1 Primary metric: PR-AUC (average precision) for `y = 1`
`y=1` is "TP touched before SL", i.e. the **profitable class** on the gross R basis. Why PR-AUC and not accuracy/ROC:

- **Accuracy is disqualified:** the base rate sits near 40–45%, so "predict everything negative" scores 55–60% accuracy and is a worthless strategy. Accuracy also changes meaning if the base rate drifts between folds.
- **ROC-AUC is threshold-free but base-rate blind:** it rewards ranking headroom in a region of the curve the strategy never trades (we only trade the top of the ranking, at a threshold that buys 1.5R targets). PR-AUC only rewards improving the precision *of the trades we would actually take*, and its no-skill reference is the prevalence itself — so its lift over prevalence is directly interpretable as filtering value.
- **Expectancy-aware, not accuracy-aware:** the primary is a ranking metric, so the tie-breaker (§6.2) is where P&L enters — and the recommendation (§6.4) requires positive net expectancy, not a pretty curve.

### 6.2 Tie-breaker: expectancy per trade net of `cost_bps` on validation
`E_net = mean(R_net)` over validation trades accepted at the validation-chosen threshold. Ties within **0.01 PR-AUC** are broken by `E_net`; ties within 0.01R by fewer features (parsimony), then by lower seed variance.

### 6.3 Full ranking, and the reporting honesty it needs
- Ranked table of **all** models on validation: PR-AUC (primary), ROC-AUC, Brier, log loss, `E_net` at chosen threshold, n trades accepted, and PR-AUC lift over prevalence. Then the same table on the hold-out, after freeze.
- **Reported caveats:** (a) validation is used for selection *and* for M6's weights — I will state that M6/M7 carry extra selection optimism and show the nested-check number beside the headline; (b) with ~5 folds, per-fold PR-AUC std is large, so I report pooled OOF PR-AUC (primary) *and* per-fold mean ± std, and I use a **paired bootstrap** for model-vs-model differences rather than comparing point estimates; (c) a difference smaller than the paired-bootstrap CI is reported as "indistinguishable", not as a win.
- **Multiplicity:** the number of tuned configurations evaluated is logged; the report includes a **Deflated-Sharpe-style / PBO-style** warning (§9.6) rather than pretending that picking the best of ~800 configurations is free.

### 6.4 Recommendation rule
Recommend the model with the best validation PR-AUC **that also shows positive net expectancy on validation** and whose baseline-relative lift has a paired-bootstrap CI excluding 0. If the simple model (M1/M2/M3) wins, **I recommend it and say so** — the ladder exists to test that possibility, not to justify complexity. If everything is indistinguishable, the recommendation is "no filter / deterministic regime filter" and the report says the ML filter was not validated.

---

## 7. Calibration and threshold protocol

### 7.1 Calibration
- **Method choice is made inside the training block**, never on validation: carve the last 20% (by time) of the outer training block as a calibration set, purge/embargo `max_hold` between fitting data and calibration data. Fit both Platt and isotonic on that split; choose by calibration-set **Brier score** (log loss reported alongside).
- **Isotonic requires `n_calib ≥ 500`** (it overfits below that and produces step functions with empty/threshold-swallowing bins); below 500 the default is Platt, and the isotonic number is still reported for transparency.
- After the method is chosen, refit the calibrator on the **full training block** (base model frozen) and log that this refit is the deployed one.
- **Reported:** reliability diagram (binned, with bin counts — a reliability diagram without counts hides that the top bins are 3 trades), Brier, log loss, ECE with bootstrap CI, and the calibration slope/intercept on validation. Uncalibrated scores are reported beside calibrated ones so the calibration gain is visible.
- Note stated in the report: for **threshold-based filtering, calibration does not change the trade set** if the threshold is also mapped through the calibrated score — what it changes is whether "0.55" means 55%.

### 7.2 Threshold selection (pre-declared rule, not eyeballed)
1. Grid `τ ∈ [0.40, 0.70]` step 0.01 on **validation** calibrated scores.
2. For each τ: accepted count, hit rate, `E_net`, total R. Bootstrap (10k, trade-level) the `E_net` curve → SE(τ).
3. `τ*` = **smallest τ whose `E_net` is within 1 SE of the grid maximum**, subject to `n_accepted ≥ 20` overall and `≥ 5` per fold. (One-standard-error rule: it deliberately trades a sliver of fitted expectancy for a threshold in a *flat* region with more trades — the argmax of a noisy curve is the classic overfit.)
4. Report the **stability region** = contiguous τ range where `E_net` is within 1 SE of max; state whether `τ*` sits inside it and how wide it is. A narrow region is reported as "threshold not robust".
5. Freeze `τ*` and the calibrator hash in `freeze.json` at G5. Report the hold-out at that frozen τ **and** the hold-out expectancy curve across the whole grid (so you can see it is not a fluke of one τ), without re-choosing.

---

## 8. Diagnostics (Phase 4) — what will prove learning vs. noise

Run for **M1, M4, M5, and the recommended model** (M1/M4/M5 as required; the winner may add a fourth/second pass).

| Diagnostic | Protocol | Pass criterion / how noise would show up |
|---|---|---|
| **Learning curves** | Refit with frozen best hyperparameters on expanding fractions (20/40/60/80/100%) of each outer training block, always expanding from the earliest bar, purged each time; report train vs validation PR-AUC and `E_net` vs n. Also the reverse direction (most-recent-only) to detect regime dependence. | Learning (not noise) = validation curve rises with n, train-val gap shrinks or stays bounded. A flat/declining validation curve at n < 1000 is the honest signature of "this dataset cannot support this model" and gets reported as such. |
| **Permutation importance** | sklearn `permutation_importance`, scoring = average precision, `n_repeats=30`, on each fold's **validation** block; report mean ΔPR-AUC, and the **rank mean ± std across folds**. | Stable ranks across folds = a real signal. Ranks that scramble across folds = the model is picking up fold-specific noise; that is the finding. |
| **SHAP** | `TreeExplainer` for M4/M5 (and `LinearExplainer`/coefficient path for M1, reported alongside); summary bar + beeswarm on validation, one representative fold documented, plus per-fold sign stability of the top 5. | Any feature that (a) dominates importance, (b) has Spearman |ρ| > 0.5 with realized R, and (c) has no economic story → flagged as a **label proxy** and investigated (the post-hoc leak check in §1.5.4). |
| **Shuffled-label null** | Permute `y` within training folds, **retrain the full pipeline** (same feature build, splits, tuning protocol, calibration) and re-measure validation PR-AUC. `n_perm = 20` for the winner, `n_perm = 50` for M1 (cheap). Plus a fast complementary check: 200 permutations against *fitted* models (no retrain) to get a tight null for the statistic. | Observed PR-AUC must sit far in the right tail of the null; report the null distribution, its mean/std, and the empirical p-value. If observed ≈ prevalence ≈ null mean, the model is fitting noise and the report says that first. |
| **Calibration** | Reliability diagram + Brier (calibrated vs uncalibrated), bootstrapped. | Calibration well off the 45° line at the traded threshold = the filter's probability is not meaningful. |
| **Threshold stability** | Full τ grid 0.40–0.70 × {hit rate, `E_net`, n accepted}, with bootstrap bands; state where `τ*` sits. | `E_net` positive only in a 0.02-wide spike and negative elsewhere → not robust, reported as such. |

---

## 9. Backtest design (Phase 5)

### 9.1 Three arms, per fold and pooled
1. **No filter** — every cross.
2. **Deterministic regime filter** — take the cross iff `eff_24 > 0.35` at the cross bar (causal: uses only bars ≤ cross).
3. **ML filter** — take the cross iff calibrated `P(y=1) > τ*`.

### 9.2 Metrics per arm × fold (+ pooled), with bootstrap CIs
trades, hit rate, **expectancy in R/trade (gross and net)**, profit factor, turnover (trades/yr), max drawdown (cumulative net R), Sharpe, total R. CIs: 10k trade-level bootstrap **and** a stationary block bootstrap (mean block ≈ 10 trades) because trades cluster in time; if the two disagree materially I report the wider one.

**Additional honest measures:**
- **Paired arm comparison:** the filter's value must be measured on the *same* trades — a paired bootstrap of `E_net(filter-accepted) − E_net(all)` and a "rejected trades" table (what the filter threw away, and what those trades earned). A filter that drops trades that were profitable is a filter that failed, and it will be visible.
- **Overlap/concurrency:** with `max_hold=24` on 1h bars, consecutive crosses can be in the market simultaneously. I report **both** an unconstrained R accounting (every accepted cross counted, the standard triple-barrier evaluation) **and** a one-position-at-a-time variant (skip signals while in a trade) — because overlap inflates Sharpe/DD in ways that vanish when a real account can only hold one position.
- **Sharpe definition stated explicitly:** per-trade Sharpe `mean(R)/std(R)`, annualized by `√(trades/year)`, labelled as a *trade-level* Sharpe, plus a daily-resampled equity Sharpe for comparison. Not blurred together.

### 9.3 Cost model (explicit, because it decides the verdict)
`cost_in_R = (cost_bps + slippage_bps)·1e−4 · 2 · entry_price / (sl_atr · ATR_cross)`, default `cost_bps=5`, `slippage_bps=0`, configurable; applied to every trade in all three arms (so the comparison is fair), and swept in §11.5.

### 9.4 Sensitivity (report, never optimize on test)
- Barrier: `tp_atr ∈ {1.0, 1.5, 2.0}` × `sl_atr ∈ {0.75, 1.0, 1.5}` × `max_hold ∈ {12, 24, 48}`.
- Regime rule: `eff_threshold ∈ {0.25, 0.35, 0.45}` × `lookback ∈ {12, 24, 48}`.
- Cost: `cost_bps ∈ {0, 5, 10, 20}`.
- Reported as grids/tables with the hold-out arm-3 numbers for reference. **No parameter is changed after seeing the hold-out.** Where the default barrier set was chosen on *validation* sensitivity, that is stated in the report (it is a selection, and selections get count).
- Note on sweep-induced label leakage: changing `max_hold`/`tp`/`sl` changes the labels, so each barrier setting is a *separate run* of the whole pipeline, not a post-hoc re-cut of one run's predictions. Sequence models and tuned hyperparameters are **not** re-tuned per setting — the sweep uses the frozen winner and frozen threshold, and that limitation is stated in the report (it makes the sweep a robustness check, not a second optimization).

### 9.5 Cost-budget note
Each barrier setting re-runs the full pipeline. To keep this tractable the sweep runs the **frozen recommended model + the two baselines**, not the entire ladder (the ladder is a model-selection exercise, the sweep is a robustness exercise). Stated in the report.

### 9.6 Second instrument
If you provide a second CSV, the full walk-forward repeats on it and the filter is "validated" only if it survives both. With one series, the report's limitation section says plainly: **"single-series result, not validated."** No amount of tuning fixes that, and I will not present one series as validation.

### 9.7 Overfitting ledger
The report records the number of configurations evaluated (models × trials × seeds × arms), and includes a **PBO / Deflated-Sharpe** flavored paragraph quantifying how much of the hold-out number could be selection luck, plus the single-touch ledger for the hold-out (timestamp, hash, one row per evaluation).

---

## 10. Report structure (`reports/report.md`)

1. **Verdict paragraph** — did the ML filter beat *both* baselines on the hold-out **after costs**? Yes/no with expectancy numbers. If no, said in the first sentence.
2. Ladder table M0–M8 × {PR-AUC, ROC-AUC, Brier, log loss, `E_net` in R} on validation and hold-out, with bootstrap CIs (skipped models shown as skipped, with reason).
3. Feature table — formula, rationale, permutation rank, fold stability, ρ with R, keep/drop reason.
4. Tuning report — search space, best params per fold, param stability, trials completed/pruned.
5. Diagnostics — learning curves, SHAP, shuffled-label null, calibration, threshold stability.
6. Three-arm backtest tables — per fold + pooled, all sensitivity sweeps, bootstrap CIs, paired comparisons.
7. **"What would make this better"** — data (tick/LOB, funding, cross-asset), features (regime conditioning, term structure of vol), labels (meta-labeling on a better primary, cost-aware labels, path-dependent barriers), execution (limit entries, slippage model, position sizing / Kelly fraction), statistics (more instruments, longer history). Not hyperparameters.
8. Reproducibility — seeds, data sha256, config sha256, plan sha256, split row counts, exact code references (file:line for feature and label functions), library versions, wall-clock.

Every requirement from your brief appears in that structure explicitly; nothing is dropped silently.

---

## 11. Engineering / reproducibility

- Package `src/dema/` with modules: `data.py`, `features.py`, `labels.py`, `splits.py`, `models/`, `tune.py`, `calibrate.py`, `evaluate.py`, `diagnostics.py`, `backtest.py`, `cli.py`.
- One CLI: `python -m dema.cli <phase|stage> --config configs/config.yaml [--smoke-test] [--data PATH]`, subcommands: `audit`, `features`, `ladder`, `diagnose`, `freeze`, `holdout`, `backtest`, `sweep`, `report`. Each writes `artifacts/<stage>/manifest.json` with seeds, hashes, row counts, versions, wall-clock.
- `configs/config.yaml`: data, timeframe, instrument, tz, ema spans, tp/sl/hold, cost/slippage, split geometry (n_folds, embargo, holdout fraction), feature toggles + tier policy, model spaces, trial counts, seeds, output paths. Config hash in every manifest.
- `requirements.txt` pinned; `make` targets for `test`, `leakage`, `ladder`, `report`.
- Tests: `test_leakage.py` (§1.5, G1), `test_labels.py` (TP/SL/time-barrier edge cases incl. same-bar and gap-through), `test_splits.py` (purge/embargo arithmetic, no time inversion), `test_costs.py` (cost-in-R formula), `test_no_shuffle_cv.py` (grep guard: no `KFold`/`StratifiedKFold`/`shuffle=True`), `test_reproducibility.py` (same seed → same metrics to 1e-9).
- Determinism: `numpy`/`random`/`optuna`/`torch` seeds set; Optuna TPE seeded; single-threaded where numerics matter; BLAS thread count pinned.

---

## 12. What I will *not* do (binding on me)

- No code before approval.
- No `scaler.fit` on the full sample, no full-sample quantiles, no target encoding, no future-bar reads in features (CI-enforced).
- No tuning, feature selection, calibration-method choice, or threshold choice on the hold-out; hold-out touched once.
- No accuracy headline, no ROC-only headline, no reporting a filter win without the paired comparison and the rejected-trades table.
- No iterating on the hold-out result to make the backtest look better. If the numbers are bad, the verdict paragraph says so first.
- No skipping a ladder rung or an ablation because it "seems unnecessary". Skips happen only when a pre-registered gate is not met (e.g. `n_crosses < 500` for M8), and are reported with the gate value.

---

## 13. Open questions for you (please answer with the approval)

1. **Data:** CSV path, instrument, source/exchange, `timestamp` timezone, date range. Any known gaps/outages I should be told about?
2. **Second instrument:** available?
3. **Costs:** is `cost_bps=5` round-trip or per side in your convention? Any maker/taker split or funding to model?
4. **M8:** install torch (CPU) — yes/no?
5. **Hold-out:** OK to define it as the most recent block with ≥ 40 crosses and ≥ 15% of crosses, boundary frozen before modeling?
6. **Proposed feature additions** (§2.8): approve A1+A2 and drop A3–A6, or a different subset?

---

## 14. Change log (post-approval edits)

| Date | Change | Reason | Approved by |
|---|---|---|---|
| — | — | — | — |
