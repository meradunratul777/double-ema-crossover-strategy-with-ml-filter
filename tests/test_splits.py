"""Split geometry: purge, embargo, no time inversion, hold-out integrity, A6 gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dema.config import load_config
from dema.data import synthetic_ohlcv
from dema.labels import BarrierSpec, detect_crosses, triple_barrier_labels
from dema.splits import SplitError, audit_table, make_splits, split_markdown


@pytest.fixture(scope="module")
def data():
    cfg = load_config("configs/config.yaml")
    df, _ = synthetic_ohlcv(n_bars=9000, seed=5)
    crosses = detect_crosses(df, warmup=200)
    lab = triple_barrier_labels(df, crosses, BarrierSpec(1.5, 1.0, 24), total_cost_bps=7.0)
    lab["ts_cross"] = df["timestamp"].to_numpy()[lab["cross_idx"].to_numpy()]
    lab = lab[lab["label_complete"]].reset_index(drop=True)
    return cfg, df, lab


def test_folds_do_not_leak_across_the_boundary(data):
    cfg, df, lab = data
    scheme = make_splits(lab, len(df), cfg)
    for f in scheme.folds:
        tr = lab.iloc[f.train_pos]
        va = lab.iloc[f.val_pos]
        # nothing in training may reach into the validation block
        assert tr["label_end_bar"].max() < f.v0
        assert tr["cross_idx"].max() < va["cross_idx"].min()
        # validation rows are strictly inside their block
        assert (va["cross_idx"] >= f.v0).all() and (va["cross_idx"] < f.v1).all()


def test_embargo_is_at_least_max_hold(data):
    cfg, df, lab = data
    scheme = make_splits(lab, len(df), cfg)
    assert scheme.embargo >= cfg["labels"]["max_hold"]
    for f in scheme.folds:
        gap = f.v0 - lab.iloc[f.train_pos]["label_end_bar"].max()
        assert gap > scheme.embargo


def test_holdout_is_most_recent_and_untouched(data):
    cfg, df, lab = data
    scheme = make_splits(lab, len(df), cfg)
    ho = lab.iloc[scheme.holdout_pos]
    assert (ho["cross_idx"] >= scheme.holdout_start_bar).all()
    assert lab.iloc[scheme.final_train_pos]["label_end_bar"].max() < scheme.holdout_start_bar
    # no fold's validation reaches the hold-out
    for f in scheme.folds:
        assert f.v1 <= scheme.holdout_start_bar
    n_min = max(cfg["splits"]["holdout_min_crosses"],
                int(np.ceil(cfg["splits"]["holdout_min_frac"] * scheme.n_crosses)))
    assert len(ho) >= n_min


def test_walk_forward_folds_are_expanding_and_ordered(data):
    cfg, df, lab = data
    scheme = make_splits(lab, len(df), cfg)
    assert len(scheme.folds) >= 4
    sizes = [len(f.train_pos) for f in scheme.folds]
    assert sizes == sorted(sizes)
    starts = [f.v0 for f in scheme.folds]
    assert starts == sorted(starts)


def test_purge_and_embargo_counts_are_reported(data):
    cfg, df, lab = data
    scheme = make_splits(lab, len(df), cfg)
    tbl = audit_table(scheme, lab)
    assert {"n_purged", "n_embargoed", "pos_val", "underpowered"}.issubset(tbl.columns)
    assert (tbl["n_purged"] >= 0).all()
    md = split_markdown(scheme, lab, "# audit")
    assert "hold-out" in md


def test_fingerprint_is_deterministic(data):
    cfg, df, lab = data
    a = make_splits(lab, len(df), cfg)
    b = make_splits(lab, len(df), cfg)
    assert a.fingerprint() == b.fingerprint()


def test_embargo_below_max_hold_is_rejected(data):
    cfg, df, lab = data
    bad = {**cfg, "splits": {**cfg["splits"], "embargo_bars": 3}}
    with pytest.raises(SplitError):
        make_splits(lab, len(df), bad)


def test_too_many_folds_raises(data):
    cfg, df, lab = data
    bad = {**cfg, "splits": {**cfg["splits"], "n_folds": 60}}
    with pytest.raises(SplitError):
        make_splits(lab, len(df), bad)
