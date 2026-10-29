"""Command line entry point.

    python -m dema.cli audit --smoke-test
    python -m dema.cli audit --data /path/to/ohlcv.csv
    python -m dema.cli make-csv --rows 3000      # write the synthetic fixture to CSV

Phases are separate subcommands so no single invocation has to survive a whole
session (plan A4: checkpointing + chunked stages).
"""
from __future__ import annotations

import argparse
import numpy as np
import json
import sys
from pathlib import Path

from .audit import run_audit
from .config import load_config


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key.path=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_audit(args) -> int:
    cfg = load_config(args.config, _parse_overrides(args.set))
    res = run_audit(
        cfg,
        smoke_test=args.smoke_test,
        data_path=args.data,
        out_dir=args.out,
        skip_leakage=args.skip_leakage,
    )
    m = res["manifest"]
    print(f"\n=== Phase 1 audit — {'SYNTHETIC - not tradeable' if m['synthetic'] else 'REAL DATA'} ===")
    print(f"bars: {m['data_audit']['n_rows']}  crosses: {m['labels']['n_crosses_detected']}  "
          f"complete labels: {m['labels']['n_complete']}")
    print(f"effective warmup: {m['warmup']['effective_warmup']} bars "
          f"({m['warmup']['pct_of_sample_in_warmup']}% of sample)")
    print(f"base rate: {m['label_summary']['base_rate']:.4f}   "
          f"breakeven (net): {m['label_summary']['breakeven_hit_rate_net']:.4f}")
    print(f"mean R gross/net: {m['label_summary']['mean_R_gross']:.4f} / "
          f"{m['label_summary']['mean_R_net']:.4f}")
    print("\nleakage gate (G1):")
    for r in m["leakage"]:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['test']}")
    print(f"\nG1 green: {m['gate_green']}")
    print(f"hold-out starts at bar {m['split_boundaries']['holdout_start_bar']} "
          f"({m['split_boundaries']['holdout_n']} crosses, "
          f"{m['split_boundaries']['holdout_pos']} positives)")
    for f in res["split_table"].itertuples():
        flag = "  <-- UNDER-POWERED" if f.underpowered else ""
        print(f"  {f.fold:8s} bars {f.bar_start:>6}-{f.bar_end:<6} "
              f"train {f.n_train:>4} val {f.n_val:>4} pos_val {f.pos_val:>3} "
              f"purged {f.n_purged:>3} embargoed {f.n_embargoed:>3}{flag}")
    print(f"\nartifacts -> {Path(args.out or cfg['run']['artifact_dir']).resolve()}")
    return 0 if m["gate_green"] else 1


def cmd_make_csv(args) -> int:
    from .data import synthetic_ohlcv

    cfg = load_config(args.config)
    df, _ = synthetic_ohlcv(n_bars=args.rows, seed=args.seed)
    out = Path(args.out or "data/synthetic_ohlcv.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} SYNTHETIC rows -> {out}")
    return 0


def cmd_features(args) -> int:
    """Phase 2: build and persist the feature matrices (no training)."""
    from .audit import build_labels, resolve_warmup
    from .data import get_data
    from .features import build_bar_features, build_cross_matrix, MODEL_POOL
    import pandas as pd

    cfg = load_config(args.config, _parse_overrides(args.set))
    df, da = get_data(cfg, smoke_test=args.smoke_test, data_path=args.data)
    if da.synthetic and not args.smoke_test:
        raise SystemExit("no real CSV: pass --smoke-test to build on SYNTHETIC data")
    warmup, winfo = resolve_warmup(df, cfg)
    labels, _ = build_labels(df, cfg, warmup)
    complete = labels[labels["label_complete"]].reset_index(drop=True)
    bar = build_bar_features(df, rank_window=int(cfg["features"]["rank_window"]),
                             ffd_d_values=tuple(cfg["features"]["ffd_d_values"]))
    # FFD order is chosen on the FIRST TRAINING BLOCK only (never validation/hold-out)
    from .features import select_ffd_d
    from .splits import make_splits as _mk
    _scheme = _mk(complete, len(df), cfg)
    ffd_info = select_ffd_d(np.log(df["close"]), _scheme.folds[0].v0,
                            d_values=tuple(cfg["features"]["ffd_d_values"]),
                            p_threshold=float(cfg["features"]["ffd_adf_p"]))
    print(f"FFD selection (train fold 0, bars < {_scheme.folds[0].v0}): "
          f"p={ {k: round(v,6) if v==v else None for k,v in ffd_info['p_values'].items()} } "
          f"-> d={ffd_info['chosen_d']} ({ffd_info['rule']})")
    out = Path(args.out or cfg["run"]["artifact_dir"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    bar.to_csv(out / "bar_features.csv", index=False)
    X = build_cross_matrix(df, bar, complete, ffd_d=ffd_info["chosen_d"])
    X.to_csv(out / "cross_features.csv", index=False)

    # ---- stationarity + label-proxy diagnostics (training block only) ------
    from scipy.stats import spearmanr
    from statsmodels.tsa.stattools import adfuller
    from .features import REGISTRY
    from .splits import make_splits

    scheme = make_splits(complete, len(df), cfg)
    first = scheme.folds[0]
    train_rows = first.train_pos
    rows = []
    for name in MODEL_POOL:
        spec = REGISTRY[name]
        vals = X[name].to_numpy(dtype=float)[train_rows]
        ok = np.isfinite(vals)
        p_adf = np.nan
        tested = False
        constant = bool(ok.sum() > 0 and np.nanstd(vals[ok]) < 1e-12)
        if spec.transform_family == "continuous" and not constant and ok.sum() > 60:
            try:
                p_adf = float(adfuller(vals[ok], maxlag=10, autolag="AIC", regresults=False)[1])
                tested = True
            except Exception:
                p_adf = np.nan
        y = complete["y"].to_numpy(dtype=float)[train_rows]
        rho = spearmanr(vals[ok], y[ok]).statistic if ok.sum() > 30 else np.nan
        rows.append({
            "feature": name, "tier": spec.tier, "scope": spec.scope, "source": spec.source,
            "window": spec.window, "n_finite_train": int(ok.sum()),
            "mean": float(np.nanmean(vals)) if ok.sum() else np.nan,
            "std": float(np.nanstd(vals)) if ok.sum() else np.nan,
            "adf_p_train": p_adf,
            "adf_tested": tested,
            "constant_in_train": constant,
            "stationary_at_5pct": bool((not tested) or (np.isfinite(p_adf) and p_adf < 0.05)),
            "spearman_with_R_gross": float(rho) if np.isfinite(rho) else np.nan,
            "label_proxy_flag": bool(np.isfinite(rho) and abs(rho) > 0.5),
            "rationale": spec.rationale,
        })
    diag = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out / "feature_diagnostics.csv", index=False)
    print(f"bar features: {bar.shape} -> {out/'bar_features.csv'}")
    print(f"cross matrix: {X.shape} ({len(MODEL_POOL)} pool features)")
    print(f"rows with any NaN in pool: {int(X[list(MODEL_POOL)].isna().any(axis=1).sum())}")
    print(f"non-stationary features (ADF p>=0.05, train fold 0): "
          f"{int((~diag['stationary_at_5pct']).sum())}/{int(diag['adf_tested'].sum())} tested "
          f"({int((~diag['adf_tested']).sum())} not testable: constant or categorical)")
    print(f"label-proxy flags (|rho|>0.5 with realized R): "
          f"{diag.loc[diag['label_proxy_flag'],'feature'].tolist()}")
    n_train = len(train_rows)
    print(f"sample-size budget at fold 0: n_train={n_train}, p_pool={len(MODEL_POOL)}, "
          f"max p at n/p>=10 = {n_train // 10}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dema", description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="Phase 1 audit + leakage gate (G1)")
    a.add_argument("--smoke-test", action="store_true")
    a.add_argument("--data", default=None)
    a.add_argument("--out", default=None)
    a.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--skip-leakage", action="store_true",
                   help="dev only: marks the gate RED instead of running it")
    a.set_defaults(func=cmd_audit)

    f = sub.add_parser("features", help="Phase 2: build feature matrices")
    f.add_argument("--smoke-test", action="store_true")
    f.add_argument("--data", default=None)
    f.add_argument("--out", default=None)
    f.add_argument("--set", action="append", default=[])
    f.set_defaults(func=cmd_features)

    c = sub.add_parser("make-csv", help="write the synthetic fixture to CSV")
    c.add_argument("--rows", type=int, default=3000)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--out", default="data/synthetic_ohlcv.csv")
    c.set_defaults(func=cmd_make_csv)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
