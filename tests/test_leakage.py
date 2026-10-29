"""G1 gate: the leakage tests must pass, and they must have the power to fail.

The negative controls are the important half of this file: a leakage detector that
cannot be made to fail is decoration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dema import leakage
from dema.data import synthetic_ohlcv
from dema.features import REGISTRY, build_bar_features
from dema.labels import detect_crosses

SRC_DIR = "src/dema"


@pytest.fixture(scope="module")
def frame():
    df, _ = synthetic_ohlcv(n_bars=4000, seed=11)
    return df


@pytest.fixture(scope="module")
def crosses(frame):
    return detect_crosses(frame, warmup=150)


def test_prefix_invariance_passes(frame):
    res = leakage.prefix_invariance(frame, [1200, 2400, 3600])
    assert res.passed, res.offenders


def test_truncation_invariance_passes(frame):
    res = leakage.truncation_invariance(frame, [1200, 2400, 3600])
    assert res.passed, res.offenders


def test_shift_by_one_passes(frame):
    res = leakage.shift_by_one(frame)
    assert res.passed, res.offenders


def test_cross_matrix_invariance_passes(frame, crosses):
    res = leakage.cross_matrix_invariance(frame, crosses, checkpoints=3)
    assert res.passed, res.offenders


def test_static_source_guards_pass():
    res = leakage.static_source_guards(SRC_DIR)
    assert res.passed, res.offenders


# ---------------------------------------------------------------------------
# negative controls — the detector must catch these
# ---------------------------------------------------------------------------
def _leaky_builder(df: pd.DataFrame, **kw) -> pd.DataFrame:
    out = build_bar_features(df, **kw)
    out["_leak_fullsample"] = (df["close"] - df["close"].mean()) / df["close"].std()
    out["_leak_future"] = df["close"].shift(-1) / df["close"]
    return out


def _future_only_builder(df: pd.DataFrame, **kw) -> pd.DataFrame:
    out = build_bar_features(df, **kw)
    out["_leak_rolling_future"] = df["close"].rolling(5, center=True, min_periods=1).mean()
    return out


def test_detector_catches_fullsample_normalisation(frame):
    res = leakage.prefix_invariance(frame, [1200, 2400], builder=_leaky_builder)
    assert not res.passed
    assert any("_leak_fullsample" in o for o in res.offenders)


def test_detector_catches_forward_shift(frame):
    res = leakage.truncation_invariance(frame, [1200, 2400], builder=_leaky_builder)
    assert not res.passed
    assert any("_leak_future" in o for o in res.offenders)


def test_detector_catches_centred_window(frame):
    res = leakage.prefix_invariance(frame, [1200, 2400], builder=_future_only_builder)
    assert not res.passed
    assert any("_leak_rolling_future" in o for o in res.offenders)


# ---------------------------------------------------------------------------
# registry invariants that back the shift test (plan A2)
# ---------------------------------------------------------------------------
def test_time_only_features_declared_invariant():
    for name, spec in REGISTRY.items():
        if spec.source == "time":
            assert spec.leak_expectation == "invariant", name
        else:
            assert spec.leak_expectation in {"changes", "invariant"}, name


def test_no_label_or_forward_columns_in_registry():
    banned = {"y", "R_gross", "R_net", "exit_idx", "exit_price", "exit_reason", "bars_held"}
    assert not (banned & set(REGISTRY))


def test_every_registered_bar_feature_is_computed(frame):
    bar = build_bar_features(frame)
    for name, spec in REGISTRY.items():
        if spec.scope == "bar":
            assert name in bar.columns, f"registered but not computed: {name}"
