"""EMA 9/21 crossover detection, triple-barrier labels, and realized R-multiples.

Contract enforced here (and asserted, not commented):
  * a signal at bar i is entered at the OPEN of bar i+1;
  * barriers are sized with ATR measured on the CROSS bar, which is known at signal
    time; the entry bar's own range is used only for barrier touches, which happen
    after entry;
  * `y = 1` iff the take-profit is touched before the stop within `max_hold` bars;
  * same-bar TP+SL touch resolves to SL (intrabar order is unknowable on 1h bars);
  * a gap through a barrier fills at the open, so a realized R can be < -1;
  * realized R is stored for the backtest only and is never a feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr_wilder, ema, sign_changes


class LabelError(RuntimeError):
    pass


@dataclass(frozen=True)
class BarrierSpec:
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    max_hold: int = 24

    @property
    def label_window_bars(self) -> int:
        """Bars from the entry bar to the time barrier, inclusive."""
        return int(self.max_hold)


def detect_crosses(
    df: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    warmup: int = 200,
) -> pd.DataFrame:
    """Crosses where sign(EMA_fast - EMA_slow) flips. Only the flip bar is a signal;
    the following bars are not re-signalled."""
    if warmup < ema_slow:
        raise LabelError("warmup must be >= ema_slow")
    fast = ema(df["close"], ema_fast)
    slow = ema(df["close"], ema_slow)
    spread = (fast - slow).to_numpy(dtype=float)
    mask = sign_changes(spread)
    mask[: int(warmup)] = False
    mask &= np.isfinite(spread)

    n = len(df)
    idx = np.flatnonzero(mask)
    # A cross at the final bar cannot be entered (no next bar open): drop and count.
    unenterable = int((idx >= n - 1).sum())
    idx = idx[idx < n - 1]

    out = pd.DataFrame(
        {
            "cross_idx": idx,
            "side": np.sign(spread[idx]).astype(int),
            "ema_fast": fast.to_numpy()[idx],
            "ema_slow": slow.to_numpy()[idx],
            "close_at_cross": df["close"].to_numpy()[idx],
        }
    )
    out.attrs["unenterable_crosses"] = unenterable
    if (out["side"] == 0).any():
        raise LabelError("zero side on a cross bar")
    # consecutive crosses (whipsaw) are kept but flagged for the sensitivity report
    if len(out) > 1:
        gaps = np.diff(out["cross_idx"].to_numpy())
        out["bars_since_prev_cross"] = np.concatenate([[np.nan], gaps]).astype(float)
    else:
        out["bars_since_prev_cross"] = np.nan
    return out.reset_index(drop=True)


def triple_barrier_labels(
    df: pd.DataFrame,
    crosses: pd.DataFrame,
    spec: BarrierSpec,
    atr: np.ndarray | None = None,
    atr_window: int = 14,
    total_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Attach entry price, barriers, exit, realized R and binary label to each cross."""
    if atr is None:
        atr = atr_wilder(df, atr_window)

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    entry_idx = crosses["cross_idx"].to_numpy() + 1
    side = crosses["side"].to_numpy().astype(int)
    atr_cross = atr[crosses["cross_idx"].to_numpy()]

    # --- structural assertions: entry is the NEXT bar's open -------------------
    if not np.all(entry_idx == crosses["cross_idx"].to_numpy() + 1):
        raise LabelError("entry index is not cross_idx + 1")
    if np.any(entry_idx >= n):
        raise LabelError("entry index past the end of the sample")
    entry = o[entry_idx]
    if not np.allclose(entry, o[crosses["cross_idx"].to_numpy() + 1]):
        raise LabelError("entry price is not the next bar's open")

    bad_atr = ~np.isfinite(atr_cross) | (atr_cross <= 0)
    m = len(entry)
    y = np.full(m, -1, dtype=int)
    R = np.full(m, np.nan)
    exit_price = np.full(m, np.nan)
    exit_idx = np.full(m, -1, dtype=int)
    reason = np.empty(m, dtype=object)
    ambiguous = np.zeros(m, dtype=bool)
    gap_slip = np.zeros(m, dtype=bool)
    complete = np.zeros(m, dtype=bool)
    bars_held = np.full(m, np.nan)

    r_unit = spec.sl_atr * atr_cross  # 1R in price terms

    for k in range(m):
        i = int(crosses["cross_idx"].iloc[k])
        e = i + 1
        s = int(side[k])
        a = atr_cross[k]
        if bad_atr[k]:
            reason[k] = "invalid_atr"
            continue
        end = i + spec.max_hold
        if end >= n:
            reason[k] = "truncated"
            continue
        complete[k] = True
        ent = entry[k]
        tp = ent + s * spec.tp_atr * a
        sl = ent - s * spec.sl_atr * a

        hit = False
        for b in range(e, end + 1):
            ob, hb, lb = o[b], h[b], lo[b]
            # gap-through: the bar opens beyond a barrier -> fill at the open
            if s > 0:
                if ob <= sl:
                    exit_price[k], exit_idx[k], reason[k] = ob, b, "sl_gap"
                    gap_slip[k] = True
                    hit = True
                    break
                if ob >= tp:
                    exit_price[k], exit_idx[k], reason[k] = ob, b, "tp_gap"
                    hit = True
                    break
            else:
                if ob >= sl:
                    exit_price[k], exit_idx[k], reason[k] = ob, b, "sl_gap"
                    gap_slip[k] = True
                    hit = True
                    break
                if ob <= tp:
                    exit_price[k], exit_idx[k], reason[k] = ob, b, "tp_gap"
                    hit = True
                    break

            touch_sl = (lb <= sl) if s > 0 else (hb >= sl)
            touch_tp = (hb >= tp) if s > 0 else (lb <= tp)
            if touch_sl and touch_tp:
                ambiguous[k] = True
                exit_price[k], exit_idx[k], reason[k] = sl, b, "sl"
                hit = True
                break
            if touch_sl:
                exit_price[k], exit_idx[k], reason[k] = sl, b, "sl"
                hit = True
                break
            if touch_tp:
                exit_price[k], exit_idx[k], reason[k] = tp, b, "tp"
                hit = True
                break

        if not hit:
            exit_price[k], exit_idx[k], reason[k] = c[end], end, "time"

        # Realized R is taken exactly (tp_atr/sl_atr or -1) for clean barrier hits so
        # float noise cannot pollute the statistics; gap and time exits are priced.
        if reason[k] == "tp":
            R[k] = spec.tp_atr / spec.sl_atr
        elif reason[k] == "sl":
            R[k] = -1.0
        else:
            R[k] = s * (exit_price[k] - ent) / r_unit[k]
        R[k] = round(float(R[k]), 10)
        y[k] = 1 if reason[k] in ("tp", "tp_gap") else 0
        bars_held[k] = exit_idx[k] - e + 1

    cost_in_R = np.where(
        np.isfinite(r_unit) & (r_unit > 0),
        (float(total_cost_bps) * 1e-4) * entry / r_unit,
        np.nan,
    )

    out = crosses.copy()
    out["entry_idx"] = entry_idx
    out["entry_price"] = entry
    out["atr_cross"] = atr_cross
    out["tp_price"] = np.where(np.isfinite(atr_cross), entry + side * spec.tp_atr * atr_cross, np.nan)
    out["sl_price"] = np.where(np.isfinite(atr_cross), entry - side * spec.sl_atr * atr_cross, np.nan)
    out["exit_idx"] = exit_idx
    out["exit_price"] = exit_price
    out["exit_reason"] = reason
    out["bars_held"] = bars_held
    out["label_complete"] = complete
    out["same_bar_ambiguous"] = ambiguous
    out["gap_slip"] = gap_slip
    out["y"] = np.where(complete, y, -1).astype(int)
    out["R_gross"] = R
    out["cost_in_R"] = cost_in_R
    out["R_net"] = R - cost_in_R
    out["label_start_bar"] = out["cross_idx"] + 1
    out["label_end_bar"] = out["cross_idx"] + int(spec.max_hold)
    return out


def label_assertions(labels: pd.DataFrame, n_bars: int, spec: BarrierSpec | None = None) -> None:
    """Invariants that must hold for every labelled cross. Called by the audit."""
    lab = labels[labels["label_complete"]].copy()
    if len(lab) == 0:
        return
    if not (lab["exit_idx"] > lab["cross_idx"]).all():
        raise LabelError("an exit occurs at or before its cross bar")
    if (lab["exit_idx"] < lab["entry_idx"]).any():
        raise LabelError("an exit occurs before the entry bar")
    if (lab["exit_idx"] > lab["label_end_bar"]).any():
        raise LabelError("an exit occurs past the time barrier")
    if not np.allclose(lab["entry_price"], labels.loc[lab.index, "entry_price"]):
        raise LabelError("entry price mismatch")
    if not set(lab["exit_reason"]).issubset({"tp", "tp_gap", "sl", "sl_gap", "time"}):
        raise LabelError("unexpected exit reason")
    tp_mask = lab["exit_reason"].isin(["tp", "tp_gap"])
    if not (lab.loc[tp_mask, "y"] == 1).all() or not (lab.loc[~tp_mask, "y"] == 0).all():
        raise LabelError("y disagrees with exit reason")
    if lab["label_end_bar"].max() > n_bars - 1:
        raise LabelError("label window runs past the end of the sample")
    if not np.isfinite(lab["R_gross"]).all():
        raise LabelError("non-finite realized R")
    if spec is not None:
        rr = spec.tp_atr / spec.sl_atr
        tol = 1e-9
        exact_tp = lab[lab["exit_reason"] == "tp"]
        exact_sl = lab[lab["exit_reason"] == "sl"]
        if not np.allclose(exact_tp["R_gross"], rr, atol=tol):
            raise LabelError("a clean TP exit does not realize tp_atr/sl_atr")
        if not np.allclose(exact_sl["R_gross"], -1.0, atol=tol):
            raise LabelError("a clean SL exit does not realize -1R")
        gap_tp = lab[lab["exit_reason"] == "tp_gap"]
        gap_sl = lab[lab["exit_reason"] == "sl_gap"]
        if len(gap_tp) and (gap_tp["R_gross"] < rr - tol).any():
            raise LabelError("a TP gap filled worse than the TP level")
        if len(gap_sl) and (gap_sl["R_gross"] > -1.0 + tol).any():
            raise LabelError("an SL gap filled better than the SL level")


def apply_spec(labels: pd.DataFrame, spec: BarrierSpec, total_cost_bps: float) -> pd.DataFrame:
    """Recost an existing label set for a different cost assumption (no relabelling)."""
    out = labels.copy()
    out["R_net"] = out["R_gross"] - (float(total_cost_bps) * 1e-4) * out["entry_price"] / (
        spec.sl_atr * out["atr_cross"]
    )
    out["cost_in_R"] = (float(total_cost_bps) * 1e-4) * out["entry_price"] / (
        spec.sl_atr * out["atr_cross"]
    )
    return out
