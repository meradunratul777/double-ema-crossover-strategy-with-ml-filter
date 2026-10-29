"""Loader integrity: cleaning rules, gap census, watermark, CSV round-trip."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dema.data import (
    SYNTHETIC_BANNER,
    DataIntegrityError,
    frame_sha256,
    load_ohlcv,
    synthetic_ohlcv,
)
from dema.features import build_bar_features


def test_synthetic_is_watermarked_and_ohlc_valid():
    df, audit = synthetic_ohlcv(n_bars=800, seed=2)
    assert audit.synthetic is True
    assert SYNTHETIC_BANNER in audit.issues
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (df["volume"] >= 0).all()


def test_csv_round_trip_and_cleaning(tmp_path):
    df, _ = synthetic_ohlcv(n_bars=700, seed=4)
    raw = df.copy()
    # inject the four classic defects
    raw.loc[10, "close"] = np.nan
    raw.loc[20, "high"] = raw.loc[20, "low"] * 0.5      # violates high >= low
    raw.loc[30, "volume"] = -5.0                        # negative volume
    dup = raw.iloc[[40]]
    raw = pd.concat([raw, dup], ignore_index=True)      # duplicate timestamp
    raw = raw.sample(frac=1.0, random_state=1)          # unsorted
    path = tmp_path / "bars.csv"
    raw.to_csv(path, index=False)

    df2, audit = load_ohlcv(path)
    assert audit.synthetic is False
    assert audit.duplicates >= 1
    assert audit.unsorted > 0
    assert audit.nan_rows >= 1
    assert audit.bad_ohlc >= 1
    assert audit.negative_volume >= 1
    assert df2["timestamp"].is_monotonic_increasing
    assert not df2.duplicated(subset=["timestamp"]).any()
    # 3 defective rows dropped; the injected duplicate nets to zero
    assert len(df2) == len(df) - 3


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [1], "close": [1.0]}).to_csv(p, index=False)
    with pytest.raises(DataIntegrityError):
        load_ohlcv(p)


def test_gap_census(tmp_path):
    df, _ = synthetic_ohlcv(n_bars=700, seed=6)
    df = df.drop(index=range(100, 140)).reset_index(drop=True)   # 40h hole
    path = tmp_path / "gappy.csv"
    df.to_csv(path, index=False)
    _, audit = load_ohlcv(path)
    assert audit.n_gaps >= 1
    assert audit.longest_gap_hours >= 40


def test_has_gap_flag_marks_window_after_gap():
    df, _ = synthetic_ohlcv(n_bars=600, seed=8)
    df = df.drop(index=range(300, 330)).reset_index(drop=True)
    bar = build_bar_features(df)
    idx = df.index[df["timestamp"].diff() > pd.Timedelta(hours=1.5)]
    assert len(idx) == 1
    after = bar["has_gap_24"].iloc[idx[0] : idx[0] + 24]
    assert after.max() == 1.0
    assert bar["has_gap_24"].iloc[-1] == 0.0


def test_frame_hash_is_stable_and_order_sensitive():
    df, _ = synthetic_ohlcv(n_bars=300, seed=9)
    assert frame_sha256(df) == frame_sha256(df.copy())
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert frame_sha256(shuffled) != frame_sha256(df)
