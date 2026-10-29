"""Configuration loading, validation, and hashing.

Every artifact manifest records the sha256 of the resolved config so a reported
number can always be traced back to the exact settings that produced it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = ("run", "data", "signal", "labels", "costs", "splits", "features")


class ConfigError(ValueError):
    pass


def load_config(path: str | Path = "configs/config.yaml", overrides: dict | None = None) -> dict:
    """Load YAML config, apply dotted-key overrides, validate, and attach metadata."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config root must be a mapping")
    cfg = copy.deepcopy(raw)
    for key, value in (overrides or {}).items():
        _set_dotted(cfg, key, value)
    validate_config(cfg)
    cfg["_meta"] = {
        "config_path": str(path),
        "config_sha256": config_hash(cfg),
    }
    return cfg


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in d:
            raise ConfigError(f"override path not present in config: {dotted}")
        d = d[k]
    if keys[-1] not in d:
        raise ConfigError(f"override path not present in config: {dotted}")
    d[keys[-1]] = value


def config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def validate_config(cfg: dict) -> None:
    for key in REQUIRED_TOP_LEVEL:
        if key not in cfg:
            raise ConfigError(f"missing required config section: {key}")

    lbl = cfg["labels"]
    if lbl["tp_atr"] <= 0 or lbl["sl_atr"] <= 0:
        raise ConfigError("tp_atr and sl_atr must be > 0")
    if int(lbl["max_hold"]) < 1:
        raise ConfigError("max_hold must be >= 1")
    if int(lbl["atr_window"]) < 2:
        raise ConfigError("atr_window must be >= 2")

    sig = cfg["signal"]
    if int(sig["ema_fast"]) >= int(sig["ema_slow"]):
        raise ConfigError("ema_fast must be < ema_slow")
    if int(sig["warmup"]) < int(sig["ema_slow"]):
        raise ConfigError("warmup must be >= ema_slow (EMA convergence)")

    costs = cfg["costs"]
    for k in ("cost_bps", "slippage_bps", "funding_bps"):
        if costs[k] < 0:
            raise ConfigError(f"costs.{k} must be >= 0")

    sp = cfg["splits"]
    if sp["embargo_bars"] is None:
        sp["embargo_bars"] = int(lbl["max_hold"])
    # Purge + embargo must be >= max_hold (plan section 1.4). Purge is implicit:
    # any training label window that touches the validation block is dropped.
    if int(sp["embargo_bars"]) < 0:
        raise ConfigError("embargo_bars must be >= 0")
    if int(sp["n_folds"]) < 1:
        raise ConfigError("n_folds must be >= 1")
    if not 0 < float(sp["min_train_frac"]) < 1:
        raise ConfigError("min_train_frac must be in (0,1)")
    if not 0 < float(sp["holdout_min_frac"]) < 1:
        raise ConfigError("holdout_min_frac must be in (0,1)")

    f = cfg["features"]
    if int(f["rank_window"]) < 2:
        raise ConfigError("rank_window must be >= 2")
    for d in f["ffd_d_values"]:
        if not 0 < float(d) <= 1:
            raise ConfigError("ffd d values must be in (0,1]")

    th = cfg.get("threshold", {})
    if th:
        lo, hi, step = th["grid"]
        if not lo < hi or step <= 0:
            raise ConfigError("threshold.grid must be [lo, hi, step] with lo < hi")


def total_cost_bps(cfg: dict) -> float:
    """Approved cost convention: cost_bps is ROUND-TRIP total, slippage is PER SIDE."""
    c = cfg["costs"]
    return float(c["cost_bps"]) + 2.0 * float(c["slippage_bps"]) + float(c.get("funding_bps", 0.0))


def resolved_embargo(cfg: dict) -> int:
    return int(cfg["splits"]["embargo_bars"])


def dump_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=True))
