# Label audit — SYNTHETIC - not tradeable

Generator: `synthetic(seed=7, n_bars=26280)`

Barriers: TP 1.5R / SL 1.0R / hold 24 bars, ATR(14) at the cross bar; entry at the NEXT bar open (asserted).

Costs: 7.0 bps round trip (5 fees + 2 x 1 slippage).

| metric | value | note |
|---|---|---|
| crosses (complete) | 1028 | labelled trades available to the model |
| base rate P(y=1) | 0.3765 | TP touched before SL |
| breakeven hit rate (gross) | 0.4 | 1/(1+tp/sl) with tp/sl=1.50 |
| breakeven hit rate (net of costs) | 0.4139 | uses median cost 0.0594R; the bar any filter must clear |
| exit tp | 386 | clean take-profit |
| exit sl | 619 | clean stop |
| exit time | 20 | neither barrier within max_hold |
| exit tp_gap | 1 | favourable gap through TP, filled at the open |
| exit sl_gap | 2 | adverse gap through SL, filled at the open |
| same-bar TP+SL rate | 0.0058 | 1h bars cannot resolve intrabar order; resolved to SL (conservative) |
| gap-slip rate | 0.0019 | fill worse than the barrier |
| mean R gross | -0.0378 | unfiltered, before costs |
| mean R net | -0.1009 | after round-trip costs + slippage |
| median cost (R) | 0.0594 | 7.0 bps round trip |
| R quantiles 5/25/50/75/95 | -1.0, -1.0, -1.0, 1.5, 1.5 | shape of the payoff distribution |
| worst R | -1.017 | gap-through stop can exceed -1R |
| best R | 1.518 | gap-through target |
| bars held mean / median / max | 5.46, 4.0, 24.0 | time-barrier behaviour |
| consecutive crosses (<=3 bars) | 158 | whipsaw signals, kept but flagged |
| overlapping hold windows | 679 | signals whose trade overlaps the next one; reported in the backtest too |
| long / short split | 514, 514 | leg balance |
| long hit rate | 0.3755 | per-leg base rate |
| short hit rate | 0.3774 | per-leg base rate |

## Label construction counts

| count | value |
|---|---|
| n_crosses_detected | 1028 |
| n_unenterable_crosses_dropped | 0 |
| n_complete | 1028 |
| n_truncated | 0 |
| n_invalid_atr | 0 |

## Warmup / fractional-difference data cost

| item | value |
|---|---|
| configured_warmup | 200 |
| ffd_weight_lengths | {'0.3': 2275, '0.4': 1458, '0.5': 927} |
| ffd_startup_bars | 2275 |
| ffd_dropped_for_sample_size | False |
| effective_warmup | 2276 |
| pct_of_sample_in_warmup | 8.66 |
