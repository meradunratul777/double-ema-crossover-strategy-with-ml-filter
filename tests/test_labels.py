"""Label correctness: entry timing, barrier resolution, R arithmetic, cost-in-R."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dema.data import synthetic_ohlcv
from dema.indicators import atr_wilder
from dema.labels import (
    BarrierSpec,
    LabelError,
    apply_spec,
    detect_crosses,
    label_assertions,
    triple_barrier_labels,
)


def make_frame(opens, highs, lows, closes, vol=None):
    n = len(opens)
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vol if vol is not None else np.full(n, 100.0),
        }
    )


def one_cross_frame(entry_open, path_highs, path_lows, path_closes, atr_value, max_hold=5, pad=True):
    """Bar 0 = cross bar (flat), bar 1 = entry bar, then the scripted path.

    With pad=True the series is extended with flat bars so that the full label window
    `[cross+1, cross+max_hold]` always exists (otherwise every label is 'truncated' and
    the test would be measuring the wrong thing).
    """
    opens = [100.0, entry_open] + list(path_closes[:-1])
    highs = [100.5] + list(path_highs)
    lows = [99.5] + list(path_lows)
    closes = [100.0] + list(path_closes)
    if pad:
        while len(opens) - 1 < max_hold:
            opens.append(closes[-1]); highs.append(closes[-1])
            lows.append(closes[-1]); closes.append(closes[-1])
    df = make_frame(opens, highs, lows, closes)
    atr = np.full(len(df), atr_value)
    crosses = pd.DataFrame(
        {"cross_idx": [0], "side": [1], "ema_fast": [100.1], "ema_slow": [100.0],
         "close_at_cross": [100.0], "bars_since_prev_cross": [np.nan]}
    )
    return df, crosses, atr


def test_entry_is_next_bar_open():
    df, crosses, atr = one_cross_frame(
        entry_open=101.0, path_highs=[101.2, 101.3], path_lows=[100.8, 100.9],
        path_closes=[101.0, 101.1], atr_value=1.0
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr)
    assert int(lab["entry_idx"].iloc[0]) == 1
    assert lab["entry_price"].iloc[0] == pytest.approx(df["open"].iloc[1])
    # cross-bar high must NOT be able to trigger the barrier: TP/SL are set from bar 1 open
    assert lab["tp_price"].iloc[0] == pytest.approx(102.5)
    assert lab["sl_price"].iloc[0] == pytest.approx(100.0)


def test_tp_first_gives_r_multiple():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.5, 101.6, 101.0], path_lows=[99.8, 99.9, 99.0],
        path_closes=[100.2, 101.4, 100.0], atr_value=1.0
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr)
    assert lab["y"].iloc[0] == 1
    assert lab["exit_reason"].iloc[0] == "tp"
    assert lab["R_gross"].iloc[0] == pytest.approx(1.5)
    assert int(lab["exit_idx"].iloc[0]) == 2


def test_same_bar_tp_and_sl_resolves_to_stop():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[101.8], path_lows=[98.9], path_closes=[100.0], atr_value=1.0
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr)
    assert bool(lab["same_bar_ambiguous"].iloc[0]) is True
    assert lab["exit_reason"].iloc[0] == "sl"
    assert lab["y"].iloc[0] == 0
    assert lab["R_gross"].iloc[0] == pytest.approx(-1.0)


def test_gap_through_stop_can_exceed_one_R():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.4, 100.2], path_lows=[99.9, 97.0], path_closes=[100.0, 97.5],
        atr_value=1.0
    )
    df.loc[2, "open"] = 98.0  # opening gap straight through the 99.0 stop
    df.loc[2, "high"] = 98.2
    df.loc[2, "low"] = 96.9
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr)
    assert lab["exit_reason"].iloc[0] == "sl_gap"
    assert lab["R_gross"].iloc[0] < -1.0
    assert bool(lab["gap_slip"].iloc[0]) is True


def test_time_barrier_exits_at_close():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.3, 100.3, 100.3], path_lows=[99.8, 99.8, 99.8],
        path_closes=[100.1, 100.2, 100.15], atr_value=1.0, max_hold=2
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 2), atr=atr)
    assert lab["exit_reason"].iloc[0] == "time"
    assert int(lab["exit_idx"].iloc[0]) == 2  # entry bar 1 .. bar 2 = 2 bars held
    assert lab["R_gross"].iloc[0] == pytest.approx(0.20)  # exit at bar 2 close = 100.2


def test_short_side_barriers():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.2, 100.3], path_lows=[99.4, 98.3], path_closes=[99.6, 98.5],
        atr_value=1.0
    )
    crosses["side"] = [-1]
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr)
    assert lab["tp_price"].iloc[0] == pytest.approx(98.5)
    assert lab["sl_price"].iloc[0] == pytest.approx(101.0)
    assert lab["y"].iloc[0] == 1


def test_cost_in_R_matches_hand_calculation():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.5, 101.6], path_lows=[99.8, 99.9],
        path_closes=[100.2, 101.4], atr_value=1.0
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr, total_cost_bps=7.0)
    expected = 7.0 * 1e-4 * 100.0 / (1.0 * 1.0)
    assert lab["cost_in_R"].iloc[0] == pytest.approx(expected)
    assert lab["R_net"].iloc[0] == pytest.approx(1.5 - expected)


def test_apply_spec_recosts_without_relabelling():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.5, 101.6], path_lows=[99.8, 99.9],
        path_closes=[100.2, 101.4], atr_value=1.0
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr, total_cost_bps=0.0)
    re = apply_spec(lab, BarrierSpec(1.5, 1.0, 5), total_cost_bps=20.0)
    assert (re["y"] == lab["y"]).all()
    assert (re["R_gross"] == lab["R_gross"]).all()
    assert re["R_net"].iloc[0] < lab["R_gross"].iloc[0]


def test_truncated_crosses_are_flagged_not_labelled():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.5], path_lows=[99.8], path_closes=[100.2],
        atr_value=1.0, pad=False
    )
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 10), atr=atr)
    assert bool(lab["label_complete"].iloc[0]) is False
    assert lab["y"].iloc[0] == -1


def test_full_series_labels_are_consistent():
    df, _ = synthetic_ohlcv(n_bars=3000, seed=3)
    crosses = detect_crosses(df, warmup=200)
    spec = BarrierSpec(1.5, 1.0, 24)
    lab = triple_barrier_labels(df, crosses, spec, atr=atr_wilder(df, 14), total_cost_bps=7.0)
    label_assertions(lab, len(df), spec)
    comp = lab[lab["label_complete"]]
    assert (comp["label_end_bar"] <= len(df) - 1).all()
    assert (comp["exit_idx"] <= comp["label_end_bar"]).all()
    assert set(comp["y"].unique()).issubset({0, 1})


def test_entry_assertion_rejects_bad_cross_index():
    df, crosses, atr = one_cross_frame(
        entry_open=100.0, path_highs=[100.5], path_lows=[99.8], path_closes=[100.2], atr_value=1.0
    )
    crosses["cross_idx"] = [len(df) - 1]  # entry would be past the end
    with pytest.raises(LabelError):
        triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 5), atr=atr)
