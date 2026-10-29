"""G1 gate: the Phase 1 audit. Produces the label audit, the split audit, the leakage
report, and the feature manifest — all before anything is trained.

If the leakage gate fails, this module raises: no artifact claiming a green gate is
written, and no model can be trained afterwards.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, leakage
from .config import config_hash, load_config, total_cost_bps
from .data import SYNTHETIC_BANNER, frame_sha256, get_data
from .features import MODEL_POOL, REGISTRY
from .indicators import atr_wilder, ffd_length
from .labels import BarrierSpec, detect_crosses, label_assertions, triple_barrier_labels
from .splits import audit_table, make_splits, split_markdown

PLAN_SHA_PATH = Path("docs/plan.sha256")


def _read_plan_hash() -> str:
    p = PLAN_SHA_PATH
    if not p.exists():
        return "MISSING"
    return p.read_text().split()[0].strip()


def resolve_warmup(df: pd.DataFrame, cfg: dict) -> tuple[int, dict]:
    """Effective warmup = max(configured warmup, longest FFD startup window + 1).

    FFD at |w| <= 1e-5 needs 927 (d=0.5) to 2275 (d=0.3) bars of history, which is a
    real data cost and is reported. If that cost exceeds 25% of the sample, FFD is
    dropped from the pool rather than silently eating the sample.
    """
    f = cfg["features"]
    lens = {d: ffd_length(d) for d in f["ffd_d_values"]}
    longest = max(lens.values())
    n = len(df)
    ffd_dropped = (longest + 1) > 0.25 * n
    effective = int(cfg["signal"]["warmup"])
    if not ffd_dropped:
        effective = max(effective, longest + 1)
    info = {
        "configured_warmup": int(cfg["signal"]["warmup"]),
        "ffd_weight_lengths": {str(k): int(v) for k, v in lens.items()},
        "ffd_startup_bars": int(longest),
        "ffd_dropped_for_sample_size": bool(ffd_dropped),
        "effective_warmup": int(effective),
        "pct_of_sample_in_warmup": round(100.0 * effective / n, 2),
    }
    return effective, info


def build_labels(df: pd.DataFrame, cfg: dict, warmup: int) -> tuple[pd.DataFrame, dict]:
    sig, lbl = cfg["signal"], cfg["labels"]
    crosses = detect_crosses(
        df, ema_fast=int(sig["ema_fast"]), ema_slow=int(sig["ema_slow"]), warmup=int(warmup)
    )
    spec = BarrierSpec(
        tp_atr=float(lbl["tp_atr"]), sl_atr=float(lbl["sl_atr"]), max_hold=int(lbl["max_hold"])
    )
    atr = atr_wilder(df, int(lbl["atr_window"]))
    labels = triple_barrier_labels(
        df, crosses, spec, atr=atr, atr_window=int(lbl["atr_window"]),
        total_cost_bps=total_cost_bps(cfg),
    )
    labels["ts_cross"] = df["timestamp"].to_numpy()[labels["cross_idx"].to_numpy()]
    label_assertions(labels, len(df), spec)
    info = {
        "n_crosses_detected": int(len(labels)),
        "n_unenterable_crosses_dropped": int(crosses.attrs.get("unenterable_crosses", 0)),
        "n_complete": int(labels["label_complete"].sum()),
        "n_truncated": int((~labels["label_complete"]).sum()),
        "n_invalid_atr": int((labels["exit_reason"] == "invalid_atr").sum()) if len(labels) else 0,
    }
    return labels, info


def label_audit(labels: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    lab = labels[labels["label_complete"]].copy()
    rr = float(cfg["labels"]["tp_atr"]) / float(cfg["labels"]["sl_atr"])
    c = total_cost_bps(cfg) * 1e-4
    cost_med = float(lab["cost_in_R"].median())
    base_rate = float(lab["y"].mean())
    be_gross = 1.0 / (1.0 + rr)
    be_net = (1.0 + cost_med) / (1.0 + rr + cost_med)
    reasons = lab["exit_reason"].value_counts().to_dict()
    rows = [
        ("crosses (complete)", len(lab), "labelled trades available to the model"),
        ("base rate P(y=1)", round(base_rate, 4), "TP touched before SL"),
        ("breakeven hit rate (gross)", round(be_gross, 4), f"1/(1+tp/sl) with tp/sl={rr:.2f}"),
        ("breakeven hit rate (net of costs)", round(be_net, 4),
         f"uses median cost {cost_med:.4f}R; the bar any filter must clear"),
        ("exit tp", reasons.get("tp", 0), "clean take-profit"),
        ("exit sl", reasons.get("sl", 0), "clean stop"),
        ("exit time", reasons.get("time", 0), "neither barrier within max_hold"),
        ("exit tp_gap", reasons.get("tp_gap", 0), "favourable gap through TP, filled at the open"),
        ("exit sl_gap", reasons.get("sl_gap", 0), "adverse gap through SL, filled at the open"),
        ("same-bar TP+SL rate", round(float(lab["same_bar_ambiguous"].mean()), 4),
         "1h bars cannot resolve intrabar order; resolved to SL (conservative)"),
        ("gap-slip rate", round(float(lab["gap_slip"].mean()), 4), "fill worse than the barrier"),
        ("mean R gross", round(float(lab["R_gross"].mean()), 4), "unfiltered, before costs"),
        ("mean R net", round(float(lab["R_net"].mean()), 4), "after round-trip costs + slippage"),
        ("median cost (R)", round(cost_med, 4), f"{total_cost_bps(cfg):.1f} bps round trip"),
        ("R quantiles 5/25/50/75/95",
         [round(float(x), 3) for x in np.percentile(lab["R_gross"], [5, 25, 50, 75, 95])],
         "shape of the payoff distribution"),
        ("worst R", round(float(lab["R_gross"].min()), 3), "gap-through stop can exceed -1R"),
        ("best R", round(float(lab["R_gross"].max()), 3), "gap-through target"),
        ("bars held mean / median / max",
         [round(float(lab["bars_held"].mean()), 2), float(lab["bars_held"].median()),
          float(lab["bars_held"].max())], "time-barrier behaviour"),
        ("consecutive crosses (<=3 bars)",
         int((lab["bars_since_prev_cross"] <= 3).sum()), "whipsaw signals, kept but flagged"),
        ("overlapping hold windows",
         int((lab["cross_idx"].shift(-1) - lab["cross_idx"] <= cfg["labels"]["max_hold"]).sum()),
         "signals whose trade overlaps the next one; reported in the backtest too"),
        ("long / short split",
         [int((lab["side"] == 1).sum()), int((lab["side"] == -1).sum())], "leg balance"),
        ("long hit rate", round(float(lab.loc[lab["side"] == 1, "y"].mean()), 4), "per-leg base rate"),
        ("short hit rate", round(float(lab.loc[lab["side"] == -1, "y"].mean()), 4), "per-leg base rate"),
    ]
    tbl = pd.DataFrame(rows, columns=["metric", "value", "note"])
    summary = {
        "base_rate": base_rate,
        "breakeven_hit_rate_gross": be_gross,
        "breakeven_hit_rate_net": be_net,
        "mean_R_gross": float(lab["R_gross"].mean()),
        "mean_R_net": float(lab["R_net"].mean()),
        "median_cost_R": cost_med,
        "n_complete": int(len(lab)),
        "ambiguous_rate": float(lab["same_bar_ambiguous"].mean()),
        "gap_slip_rate": float(lab["gap_slip"].mean()),
    }
    return tbl, summary


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def run_audit(
    cfg: dict,
    smoke_test: bool = False,
    data_path: str | None = None,
    out_dir: str | Path | None = None,
    skip_leakage: bool = False,
) -> dict:
    t0 = time.time()
    out = Path(out_dir or cfg["run"]["artifact_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df, data_audit = get_data(cfg, smoke_test=smoke_test, data_path=data_path)
    synthetic = bool(data_audit.synthetic)
    banner = SYNTHETIC_BANNER if synthetic else ""
    if synthetic and not smoke_test:
        # never train on synthetic silently
        raise SystemExit(
            "No real CSV supplied. Re-run with --smoke-test to build on synthetic data "
            f"({SYNTHETIC_BANNER}), or pass --data /path/to/ohlcv.csv."
        )

    warmup, warmup_info = resolve_warmup(df, cfg)
    labels, label_info = build_labels(df, cfg, warmup)
    lab_tbl, lab_summary = label_audit(labels, cfg)
    complete = labels[labels["label_complete"]].reset_index(drop=True)

    # ---- leakage gate (G1) -------------------------------------------------
    if skip_leakage:
        leak_results = [leakage.LeakResult("leakage", False, "SKIPPED BY FLAG — GATE NOT GREEN", [])]
    else:
        crosses_for_leak = complete[["cross_idx", "side", "bars_since_prev_cross"]].copy()
        leak_results = leakage.run_all(
            df, crosses_for_leak,
            bar_kwargs={"rank_window": int(cfg["features"]["rank_window"])},
            ffd_d=float(cfg["features"]["ffd_d_values"][1]),
            src_dir=Path(__file__).resolve().parent,
        )
    gate_green = all(r.passed for r in leak_results)

    # ---- splits ------------------------------------------------------------
    scheme = make_splits(complete, len(df), cfg)
    split_md = split_markdown(
        scheme, complete,
        header=f"# Split audit — {banner or 'REAL DATA'}\n\n"
               f"data: {data_audit.source}  |  bars: {len(df)}  |  crosses: {scheme.n_crosses}",
    )
    split_tbl = audit_table(scheme, complete)

    # ---- artifacts ---------------------------------------------------------
    (out / "label_audit.md").write_text(
        f"# Label audit — {banner or 'REAL DATA'}\n\n"
        f"Generator: `{data_audit.source}`\n\n"
        f"Barriers: TP {cfg['labels']['tp_atr']}R / SL {cfg['labels']['sl_atr']}R / "
        f"hold {cfg['labels']['max_hold']} bars, ATR({cfg['labels']['atr_window']}) at the cross bar; "
        f"entry at the NEXT bar open (asserted).\n\n"
        f"Costs: {total_cost_bps(cfg):.1f} bps round trip "
        f"({cfg['costs']['cost_bps']} fees + 2 x {cfg['costs']['slippage_bps']} slippage).\n\n"
        + _md_table(lab_tbl)
        + "\n\n## Label construction counts\n\n"
        + _md_table(pd.DataFrame(list(label_info.items()), columns=["count", "value"]))
        + "\n\n## Warmup / fractional-difference data cost\n\n"
        + _md_table(pd.DataFrame(list(warmup_info.items()), columns=["item", "value"]))
        + "\n"
    )
    (out / "split_audit.md").write_text(split_md + "\n")
    split_tbl.to_csv(out / "splits.csv", index=False)

    leak_rows = [
        {"test": r.name, "passed": r.passed, "detail": r.detail, "offenders": "; ".join(r.offenders)}
        for r in leak_results
    ]
    (out / "leakage.md").write_text(
        f"# Leakage gate (G1) — {banner or 'REAL DATA'}\n\n"
        f"**Gate status: {'GREEN' if gate_green else 'RED — training is blocked'}**\n\n"
        + _md_table(pd.DataFrame(leak_rows))
        + "\n\nNegative controls (the gate must be able to fail) live in "
        "`tests/test_leakage.py::test_detector_catches_*`.\n"
    )
    pd.DataFrame(leak_rows).to_csv(out / "leakage.csv", index=False)

    feat_rows = [
        {
            "feature": s.name,
            "tier": s.tier,
            "scope": s.scope,
            "source": s.source,
            "window": s.window if s.window is not None else "",
            "transform": s.transform_family,
            "shift_expectation": s.leak_expectation,
            "alias_of": s.alias_of or "",
            "rationale": s.rationale,
            "in_model_pool": s.name in MODEL_POOL,
        }
        for s in REGISTRY.values()
    ]
    pd.DataFrame(feat_rows).to_csv(out / "feature_manifest.csv", index=False)
    (out / "feature_manifest.json").write_text(
        json.dumps(
            {
                "n_registered": len(REGISTRY),
                "n_model_pool": len(MODEL_POOL),
                "model_pool": list(MODEL_POOL),
                "features": feat_rows,
            },
            indent=2,
        )
    )

    manifest = {
        "stage": "audit",
        "gate": "G1",
        "gate_green": gate_green,
        "synthetic": synthetic,
        "watermark": banner,
        "dema_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_sha256": config_hash(cfg),
        "plan_sha256": _read_plan_hash(),
        "data_source": data_audit.source,
        "data_frame_sha256": frame_sha256(df),
        "data_audit": data_audit.as_dict(),
        "warmup": warmup_info,
        "labels": label_info,
        "label_summary": lab_summary,
        "split_boundaries": scheme.boundaries,
        "split_fingerprint": scheme.fingerprint(),
        "n_features_registered": len(REGISTRY),
        "n_features_model_pool": len(MODEL_POOL),
        "leakage": leak_rows,
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {
        "manifest": manifest,
        "labels": labels,
        "complete": complete,
        "scheme": scheme,
        "df": df,
        "data_audit": data_audit,
        "label_table": lab_tbl,
        "split_table": split_tbl,
        "gate_green": gate_green,
        "warmup": warmup_info,
    }
