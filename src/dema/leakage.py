"""Leakage detection. This module is the G1 gate: nothing trains unless these pass.

Four mechanical tests (plan section 1.5):

  1. `prefix_invariance`  — replace all bars after `t` with noise, recompute the whole
     feature matrix, and require the rows at index <= t to be bit-identical. This is
     the strongest available statement of "no feature reads a future bar": it catches
     full-sample normalisation, `adjust=True`, non-causal quantiles, centred windows,
     and any accidental use of the complete series.
  2. `shift_by_one` — shift close by one bar and require the *declared* expectation per
     feature (plan A2): close/volume/mixed features must move, time-only features must
     be exactly invariant, compared at the SAME bar index (never "at the cross bars",
     which move under the shift and would make the test pass or fail for the wrong
     reason).
  3. `truncation_invariance` — recompute features on the truncated frame `df[:t+1]` and
     require the value at `t` to match the full-frame value. This is the mechanical
     form of "max source bar <= t", stated per bar rather than by instrumenting reads.
  4. `static_source_guards` — reject negative shifts and centred rolling windows in the
     feature modules by AST inspection, so a future edit cannot quietly introduce one.

Test 4 is cheap insurance; 1 and 3 are the load-bearing ones.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REQUIRED_COLUMNS
from .features import REGISTRY, build_bar_features, build_cross_matrix


@dataclass
class LeakResult:
    name: str
    passed: bool
    detail: str
    offenders: list[str]


def _noise_frame(df: pd.DataFrame, start: int, seed: int = 12345) -> pd.DataFrame:
    """Same shape/timestamps, but every bar from `start` onward is random-walk noise.
    Bars before `start` are untouched."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    if start >= n - 1:
        return out
    k = n - start
    base = float(df["close"].iloc[start - 1]) if start > 0 else float(df["close"].iloc[0])
    steps = rng.standard_normal(k) * 0.01
    close = base * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[base], close[:-1]]) if k > 1 else np.array([base])
    spread = np.abs(rng.standard_normal(k)) * 0.005
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) / (1 + spread)
    vol = rng.uniform(1.0, 1e4, k)
    sl = slice(start, n)
    out.loc[out.index[sl], "close"] = close
    out.loc[out.index[sl], "open"] = open_
    out.loc[out.index[sl], "high"] = high
    out.loc[out.index[sl], "low"] = low
    out.loc[out.index[sl], "volume"] = vol
    return out


def _perm_frame(df: pd.DataFrame, start: int, seed: int = 999) -> pd.DataFrame:
    """Alternative corruption: randomly permute the future values (keeps marginals)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    k = len(out) - start
    if k <= 1:
        return out
    for col in ("open", "high", "low", "close", "volume"):
        vals = out[col].to_numpy()[:].copy()
        fut = vals[start:]
        rng.shuffle(fut)
        vals[start:] = fut
        out[col] = vals
    # restore OHLC identities so the frame stays valid
    o, c = out["open"].to_numpy(), out["close"].to_numpy()
    out["high"] = np.maximum.reduce([out["high"].to_numpy(), o, c])
    out["low"] = np.minimum.reduce([out["low"].to_numpy(), o, c])
    return out


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, name: str) -> tuple[bool, list[str]]:
    offenders = []
    if list(a.columns) != list(b.columns):
        return False, ["column sets differ"]
    for col in a.columns:
        x = a[col].to_numpy(dtype=float)
        y = b[col].to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = ~((x == y) | both_nan)
        if diff.any():
            offenders.append(f"{name}.{col} differs at {int(diff.sum())} bars")
    return (len(offenders) == 0), offenders


def prefix_invariance(
    df: pd.DataFrame,
    checkpoints: list[int],
    bar_kwargs: dict | None = None,
    verbose: bool = False,
    builder=None,
) -> LeakResult:
    """Test 1: features at bars <= t must not change when strictly-future bars are
    replaced. Bar t itself is left intact, so a difference at t is a real leak."""
    kw = bar_kwargs or {}
    build = builder or build_bar_features
    base = build(df, **kw)
    offenders: list[str] = []
    for t in checkpoints:
        for corrupt, tag in ((_noise_frame, "noise"), (_perm_frame, "permute")):
            alt = build(corrupt(df, t + 1), **kw)
            ok, bad = _frames_equal(base.iloc[: t + 1], alt.iloc[: t + 1], f"t={t}:{tag}")
            if not ok:
                offenders.extend(bad)
        if verbose:
            print(f"  prefix t={t}: {'ok' if not offenders else 'FAIL'}")
    return LeakResult(
        "prefix_invariance",
        len(offenders) == 0,
        f"checked {len(checkpoints)} cut points x 2 corruption modes "
        f"({len(base.columns)} bar features); "
        + ("future bars never influence past features" if not offenders else "LEAK"),
        offenders[:20],
    )


def truncation_invariance(
    df: pd.DataFrame, checkpoints: list[int], bar_kwargs: dict | None = None, builder=None
) -> LeakResult:
    """Test 3: f(df[:t+1])[t] == f(df)[t] for all features."""
    kw = bar_kwargs or {}
    build = builder or build_bar_features
    full = build(df, **kw)
    offenders: list[str] = []
    for t in checkpoints:
        part = build(df.iloc[: t + 1].reset_index(drop=True), **kw)
        a = full.iloc[[t]].reset_index(drop=True)
        b = part.iloc[[t]].reset_index(drop=True)
        ok, bad = _frames_equal(a, b, f"t={t}")
        if not ok:
            offenders.extend(bad)
    return LeakResult(
        "truncation_invariance",
        len(offenders) == 0,
        f"checked {len(checkpoints)} cut points: each feature's value at bar t is "
        "reproduced from bars <= t alone",
        offenders[:20],
    )


def shift_by_one(
    df: pd.DataFrame,
    checkpoints: list[int] | None = None,
    bar_kwargs: dict | None = None,
    min_change_frac: float = 0.5,
    builder=None,
) -> LeakResult:
    """Test 2 (plan A2): shift close by one bar; compare at the SAME bar index and
    assert the declared per-feature expectation."""
    kw = bar_kwargs or {}
    shifted_df = df.copy()
    shifted_df["close"] = df["close"].shift(1)
    shifted_df["open"] = df["open"].shift(1)
    shifted_df["high"] = df["high"].shift(1)
    shifted_df["low"] = df["low"].shift(1)
    shifted_df.loc[shifted_df.index[0], ["open", "high", "low", "close"]] = df.loc[
        df.index[0], ["open", "high", "low", "close"]
    ].to_numpy()

    build = builder or build_bar_features
    a = build(df, **kw)
    b = build(shifted_df, **kw)
    lo = min(checkpoints) if checkpoints else max(1000, len(df) // 10)

    offenders: list[str] = []
    rows: list[dict] = []
    for name, spec in REGISTRY.items():
        if spec.scope != "bar" or name not in a.columns:
            continue
        x = a[name].to_numpy(dtype=float)[lo:]
        y = b[name].to_numpy(dtype=float)[lo:]
        finite = np.isfinite(x) & np.isfinite(y)
        n_fin = int(finite.sum())
        if n_fin == 0:
            rows.append({"feature": name, "expectation": spec.leak_expectation, "changed_frac": np.nan})
            continue
        changed = float(np.mean(x[finite] != y[finite]))
        rows.append({"feature": name, "expectation": spec.leak_expectation, "changed_frac": changed})
        if spec.leak_expectation == "invariant" and changed > 0.0:
            offenders.append(f"{name}: time-only feature moved on a close shift ({changed:.1%})")
        if spec.leak_expectation == "changes" and changed < min_change_frac:
            offenders.append(
                f"{name}: declared close/volume-derived but only {changed:.1%} of bars changed "
                f"(< {min_change_frac:.0%}) — either misdeclared or not reading its inputs"
            )
    detail = (
        f"{len(rows)} bar features checked at the same index: "
        f"{(pd.DataFrame(rows).get('expectation') == 'invariant').sum() if rows else 0} time-only "
        "features asserted exactly invariant, the rest asserted to move"
    )
    res = LeakResult("shift_by_one", len(offenders) == 0, detail, offenders)
    res.rows = rows  # type: ignore[attr-defined]
    return res


def cross_matrix_invariance(
    df: pd.DataFrame, crosses: pd.DataFrame, checkpoints: int = 3, bar_kwargs: dict | None = None,
    ffd_d: float | None = None,
) -> LeakResult:
    """Prefix invariance applied to the cross-level model matrix: truncating the bar
    series after a cross must not change that cross's feature row."""
    kw = bar_kwargs or {}
    bar = build_bar_features(df, **kw)
    full = build_cross_matrix(df, bar, crosses, ffd_d=ffd_d, rank_window=kw.get("rank_window", 72))
    offenders: list[str] = []
    n = len(crosses)
    picks = [int(n * f) for f in np.linspace(0.3, 0.9, checkpoints)]
    for k in picks:
        t = int(crosses["cross_idx"].iloc[k])
        sub = df.iloc[: t + 1].reset_index(drop=True)
        sub_bar = build_bar_features(sub, **kw)
        sub_crosses = crosses.iloc[: k + 1].reset_index(drop=True)
        row = build_cross_matrix(sub, sub_bar, sub_crosses, ffd_d=ffd_d,
                                 rank_window=kw.get("rank_window", 72))
        ok, bad = _frames_equal(full.iloc[[k]].reset_index(drop=True),
                                row.iloc[[k]].reset_index(drop=True), f"cross k={k}")
        if not ok:
            offenders.extend(bad)
    return LeakResult(
        "cross_matrix_invariance",
        len(offenders) == 0,
        f"checked {len(picks)} cross rows: truncating the series after the cross leaves "
        "that cross's features unchanged",
        offenders[:20],
    )


FORBIDDEN_CALLS = {"shift", "rolling", "expanding", "ewm"}


def static_source_guards(src_dir: str | Path) -> LeakResult:
    """Test 4 (AST): no negative shifts, no centred rolling/expanding windows, and no
    `adjust=True` EMA in the feature/indicator modules."""
    src = Path(src_dir)
    offenders: list[str] = []
    files = [p for p in src.rglob("*.py") if p.name in {"features.py", "indicators.py"}]
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = getattr(fn, "attr", getattr(fn, "id", ""))
            if fname in FORBIDDEN_CALLS:
                for kw in node.keywords:
                    if kw.arg == "center" and getattr(kw.value, "value", False):
                        offenders.append(f"{path.name}:{node.lineno} centered rolling window")
                    if kw.arg == "adjust" and getattr(kw.value, "value", False):
                        offenders.append(f"{path.name}:{node.lineno} ewm(adjust=True)")
                if fname in {"shift", "rolling", "expanding"} and node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.UnaryOp) and isinstance(a0.op, ast.USub):
                        offenders.append(f"{path.name}:{node.lineno} negative {fname}(...) — lookahead")
    return LeakResult(
        "static_source_guards",
        len(offenders) == 0,
        f"AST scan of {len(files)} modules for negative shifts, centred windows, adjust=True",
        offenders[:20],
    )


def run_all(
    df: pd.DataFrame, crosses: pd.DataFrame | None = None, bar_kwargs: dict | None = None,
    ffd_d: float | None = None, src_dir: str | Path | None = None,
) -> list[LeakResult]:
    n = len(df)
    checkpoints = [max(300, int(n * f)) for f in (0.2, 0.4, 0.6, 0.8, 0.95)]
    results = [
        prefix_invariance(df, checkpoints, bar_kwargs),
        truncation_invariance(df, checkpoints, bar_kwargs),
        shift_by_one(df, bar_kwargs=bar_kwargs),
    ]
    if crosses is not None and len(crosses) > 10:
        results.append(cross_matrix_invariance(df, crosses, bar_kwargs=bar_kwargs, ffd_d=ffd_d))
    if src_dir is not None:
        results.append(static_source_guards(src_dir))
    return results
