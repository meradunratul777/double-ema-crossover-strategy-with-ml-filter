"""Chronological split geometry with purge + embargo, and the G1 split audit.

Rules (plan section 1.4 / A6):
  * expanding-window walk-forward on the pre-hold-out range, n_folds >= 4;
  * a training cross is usable only if its whole label window ends strictly before
    `val_start - embargo`, which simultaneously purges overlap and applies the embargo;
  * the hold-out is the most recent contiguous block holding >= max(min_crosses,
    min_frac * n_crosses) crosses, its boundary fixed before any modeling;
  * purged / embargoed counts and the minimum-positives gates are reported, never
    silently applied.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import resolved_embargo


class SplitError(RuntimeError):
    pass


@dataclass
class SplitAudit:
    fold: str
    bar_start: int
    bar_end: int
    ts_start: str
    ts_end: str
    n_train: int
    n_val: int
    n_purged: int
    n_embargoed: int
    pos_train: int
    pos_val: int
    min_pos_ok: bool
    underpowered: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Fold:
    name: str
    train_pos: np.ndarray  # positional indices into the labels frame
    val_pos: np.ndarray
    v0: int
    v1: int
    audit: SplitAudit


@dataclass
class SplitScheme:
    folds: list[Fold]
    holdout_pos: np.ndarray
    holdout_start_bar: int
    final_train_pos: np.ndarray
    embargo: int
    n_crosses: int
    boundaries: dict

    def fingerprint(self) -> str:
        payload = {
            "holdout_start_bar": int(self.holdout_start_bar),
            "embargo": int(self.embargo),
            "folds": [
                {
                    "name": f.name,
                    "v0": int(f.v0),
                    "v1": int(f.v1),
                    "train": [int(x) for x in f.train_pos],
                    "val": [int(x) for x in f.val_pos],
                }
                for f in self.folds
            ],
            "holdout": [int(x) for x in self.holdout_pos],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _not_overlapping(labels: pd.DataFrame, cutoff: int) -> np.ndarray:
    """True where the whole label window ends strictly before `cutoff`."""
    return labels["label_end_bar"].to_numpy() < cutoff


def make_splits(labels: pd.DataFrame, n_bars: int, cfg: dict) -> SplitScheme:
    """Build walk-forward folds + hold-out. `labels` must contain complete labels only."""
    sp = cfg["splits"]
    embargo = resolved_embargo(cfg)
    if embargo < int(cfg["labels"]["max_hold"]):
        raise SplitError(
            f"embargo ({embargo}) must be >= max_hold ({cfg['labels']['max_hold']}) — "
            "purge + embargo must cover the label horizon"
        )
    lab = labels[labels["label_complete"]].reset_index(drop=True)
    n = len(lab)
    if n < 60:
        raise SplitError(f"only {n} complete crosses; need >= 60 for a meaningful split")

    cross_idx = lab["cross_idx"].to_numpy()
    lab_end = lab["label_end_bar"].to_numpy()
    # ---- hold-out: most recent block with enough crosses ---------------------
    k_hold = max(int(sp["holdout_min_crosses"]), int(math.ceil(float(sp["holdout_min_frac"]) * n)))
    if k_hold >= n - 60:
        raise SplitError(f"hold-out needs {k_hold} of {n} crosses; too few crosses remain")
    holdout_start_bar = int(cross_idx[n - k_hold])
    holdout_mask = cross_idx >= holdout_start_bar
    holdout_pos = np.flatnonzero(holdout_mask)
    pre_mask = cross_idx < holdout_start_bar
    pre_pos = np.flatnonzero(pre_mask)

    # ---- walk-forward folds on the pre-hold-out range ------------------------
    first_train_end = int(round(float(sp["min_train_frac"]) * holdout_start_bar))
    if first_train_end >= holdout_start_bar - 10:
        raise SplitError("min_train_frac leaves no room for validation folds")
    n_folds = int(sp["n_folds"])
    fold_span = (holdout_start_bar - first_train_end) / n_folds
    if fold_span * 2 < cfg["labels"]["max_hold"]:
        raise SplitError("validation folds are shorter than the label horizon; reduce n_folds")

    folds: list[Fold] = []
    for f in range(n_folds):
        v0 = int(round(first_train_end + f * fold_span))
        v1 = int(round(first_train_end + (f + 1) * fold_span)) if f < n_folds - 1 else holdout_start_bar
        val_mask = (cross_idx >= v0) & (cross_idx < v1)
        # A cross is usable for training only if its label window ends strictly
        # before the embargo line. Crosses excluded are split into two reported
        # categories: PURGED (label window reaches into the validation block) and
        # EMBARGOED (window ends before the block but inside the embargo gap).
        pre_val = cross_idx < v0
        overlap_val = lab_end >= v0
        inside_embargo = (lab_end >= v0 - embargo) & (lab_end < v0)
        train_mask = pre_val & ~overlap_val & ~inside_embargo
        purged = pre_val & overlap_val
        embargoed = pre_val & inside_embargo
        train_pos = np.flatnonzero(train_mask)
        val_pos = np.flatnonzero(val_mask)
        if len(val_pos) == 0 or len(train_pos) == 0:
            raise SplitError(f"fold {f}: empty train or validation block")
        pos_tr = int(lab.loc[train_pos, "y"].sum())
        pos_va = int(lab.loc[val_pos, "y"].sum())
        min_pos = int(sp["min_pos_outer"])
        notes = []
        if pos_va < min_pos:
            notes.append(f"under-powered: {pos_va} positives < gate {min_pos} (A6)")
        audit = SplitAudit(
            fold=f"f{f}",
            bar_start=int(v0),
            bar_end=int(v1),
            ts_start=str(lab.loc[val_pos[0], "ts_cross"]) if "ts_cross" in lab else "",
            ts_end=str(lab.loc[val_pos[-1], "ts_cross"]) if "ts_cross" in lab else "",
            n_train=len(train_pos),
            n_val=len(val_pos),
            n_purged=int(purged.sum()),
            n_embargoed=int(embargoed.sum()),
            pos_train=pos_tr,
            pos_val=pos_va,
            min_pos_ok=pos_va >= min_pos,
            underpowered=pos_va < min_pos,
            notes=notes,
        )
        folds.append(Fold(f"f{f}", train_pos, val_pos, v0, v1, audit))

    # ---- final model slice: everything strictly usable before the hold-out ---
    final_cut = holdout_start_bar - embargo
    final_train_mask = _not_overlapping(lab, final_cut)
    final_train_pos = np.flatnonzero(final_train_mask)
    pos_hold = int(lab.loc[holdout_pos, "y"].sum())
    if pos_hold < int(sp["min_pos_holdout"]):
        raise SplitError(
            f"hold-out has {pos_hold} positives < gate {sp['min_pos_holdout']} (A6). "
            f"The sample supports {n} crosses; the declared hold-out rule needs "
            f">= {k_hold} crosses with >= {sp['min_pos_holdout']} positives. Provide a longer "
            "history, lower splits.holdout_min_crosses/holdout_min_frac (and say so in the "
            "report), or run with fewer folds. No silent shrinking: the gate is the gate."
        )

    # ---- no-time-inversion assertions ---------------------------------------
    for f in folds:
        if lab.loc[f.train_pos, "label_end_bar"].max() >= f.v0:
            raise SplitError(f"{f.name}: training label window reaches the validation block")
        if lab.loc[f.train_pos, "cross_idx"].max() >= lab.loc[f.val_pos, "cross_idx"].min():
            raise SplitError(f"{f.name}: training crosses are not strictly before validation")
    if lab.loc[final_train_pos, "label_end_bar"].max() >= holdout_start_bar:
        raise SplitError("final training set overlaps the hold-out")

    boundaries = {
        "n_bars": int(n_bars),
        "n_crosses_complete": int(n),
        "holdout_start_bar": int(holdout_start_bar),
        "holdout_n": int(len(holdout_pos)),
        "holdout_pos": int(pos_hold),
        "final_train_n": int(len(final_train_pos)),
        "embargo_bars": int(embargo),
        "n_folds": int(n_folds),
        "first_train_end_bar": int(first_train_end),
    }
    scheme = SplitScheme(
        folds=folds,
        holdout_pos=holdout_pos,
        holdout_start_bar=holdout_start_bar,
        final_train_pos=final_train_pos,
        embargo=embargo,
        n_crosses=n,
        boundaries=boundaries,
    )
    # hold-out single-touch ledger starts here
    scheme.boundaries["split_fingerprint"] = scheme.fingerprint()
    return scheme


def audit_table(scheme: SplitScheme, labels: pd.DataFrame) -> pd.DataFrame:
    rows = [f.audit.as_dict() for f in scheme.folds]
    lab = labels[labels["label_complete"]].reset_index(drop=True)
    pos_hold = int(lab.loc[scheme.holdout_pos, "y"].sum())
    rows.append(
        SplitAudit(
            fold="holdout",
            bar_start=int(scheme.holdout_start_bar),
            bar_end=int(scheme.boundaries["n_bars"]) - 1,
            ts_start=str(lab.loc[scheme.holdout_pos[0], "ts_cross"]) if "ts_cross" in lab else "",
            ts_end=str(lab.loc[scheme.holdout_pos[-1], "ts_cross"]) if "ts_cross" in lab else "",
            n_train=len(scheme.final_train_pos),
            n_val=len(scheme.holdout_pos),
            n_purged=int(
                (
                    (lab["cross_idx"] < scheme.holdout_start_bar)
                    & (lab["label_end_bar"] >= scheme.holdout_start_bar - scheme.embargo)
                ).sum()
            ),
            n_embargoed=0,
            pos_train=int(lab.loc[scheme.final_train_pos, "y"].sum()),
            pos_val=pos_hold,
            min_pos_ok=pos_hold >= 15,
            underpowered=pos_hold < 15,
            notes=["hold-out: single touch, only after freeze.json"],
        ).as_dict()
    )
    return pd.DataFrame(rows)


def split_markdown(scheme: SplitScheme, labels: pd.DataFrame, header: str) -> str:
    df = audit_table(scheme, labels)
    lines = [header, "", "## Split geometry", "", f"- embargo: {scheme.embargo} bars",
             f"- complete crosses: {scheme.n_crosses}",
             f"- hold-out starts at bar {scheme.holdout_start_bar} "
             f"({len(scheme.holdout_pos)} crosses, {int(labels.loc[scheme.holdout_pos,'y'].sum())} positives)",
             f"- split fingerprint: `{scheme.fingerprint()}`", "", "| fold | bars | val n | pos val | train n | pos train | purged | embargoed | under-powered |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['fold']} | {r['bar_start']}–{r['bar_end']} | {r['n_val']} | {r['pos_val']} | "
            f"{r['n_train']} | {r['pos_train']} | {r['n_purged']} | {r['n_embargoed']} | "
            f"{'YES' if r['underpowered'] else 'no'} |"
        )
    return "\n".join(lines)
