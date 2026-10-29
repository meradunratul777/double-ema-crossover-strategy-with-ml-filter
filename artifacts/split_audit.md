# Split audit — SYNTHETIC - not tradeable

data: synthetic(seed=7, n_bars=26280)  |  bars: 26280  |  crosses: 1028

## Split geometry

- embargo: 24 bars
- complete crosses: 1028
- hold-out starts at bar 22525 (155 crosses, 53 positives)
- split fingerprint: `91aa5d1148ae1750c0c670cb7a61e96f7e10b381b7c200f863ad23ee97f5a080`

| fold | bars | val n | pos val | train n | pos train | purged | embargoed | under-powered |
|---|---|---|---|---|---|---|---|---|
| f0 | 7884–10812 | 111 | 45 | 261 | 83 | 2 | 0 | no |
| f1 | 10812–13740 | 120 | 47 | 371 | 127 | 2 | 1 | no |
| f2 | 13740–16669 | 120 | 54 | 490 | 175 | 2 | 2 | no |
| f3 | 16669–19597 | 135 | 54 | 613 | 229 | 1 | 0 | no |
| f4 | 19597–22525 | 124 | 50 | 748 | 283 | 0 | 1 | no |
| holdout | 22525–26279 | 155 | 53 | 870 | 333 | 3 | 0 | no |
