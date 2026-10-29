"""Guard rails: no time-shuffling splitters anywhere (plan A8).

Forbidden: KFold, StratifiedKFold, GroupKFold>ShuffleSplit variants used on time series,
and any `shuffle=True` passed to a splitter.
Explicitly ALLOWED: sklearn.inspection.permutation_importance — it permutes feature
values *within* a fold and never reorders time. The earlier blanket wording would have
false-failed CI on the Phase 4 diagnostics.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("src")
FORBIDDEN = re.compile(r"\b(KFold|StratifiedKFold|GroupKFold|ShuffleSplit|StratifiedShuffleSplit|RepeatedKFold)\s*\(")
SHUFFLE_TRUE = re.compile(r"shuffle\s*=\s*True")
ALLOWED_IMPORT = "permutation_importance"


def _sources() -> list[Path]:
    if not SRC.exists():
        return []
    return [p for p in SRC.rglob("*.py")]


def test_no_shuffling_splitter_in_source():
    offenders = []
    for path in _sources():
        text = path.read_text()
        if "test_guards" in path.name:
            continue
        for m in FORBIDDEN.finditer(text):
            offenders.append(f"{path}:{m.group(0)}")
        for m in SHUFFLE_TRUE.finditer(text):
            offenders.append(f"{path}: shuffle=True")
    assert not offenders, offenders


def test_permutation_importance_is_explicitly_allowed():
    """Documented whitelist: a diagnostic that permutes feature values is not a
    time-shuffle. This test exists so the guard above cannot be widened by accident."""
    assert ALLOWED_IMPORT not in FORBIDDEN.pattern


def test_no_accuracy_as_selection_metric():
    """Selection must be PR-AUC / expectancy driven; `accuracy_score` may appear only
    in reporting helpers, never in a selection objective."""
    offenders = []
    for path in _sources():
        text = path.read_text()
        for line in text.splitlines():
            if "accuracy" in line and ("score_func" in line or "objective" in line or "refit" in line):
                offenders.append(f"{path}: {line.strip()}")
    assert not offenders, offenders
