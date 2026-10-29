"""Feature engineering — every feature is declared in a registry with its formula,
window, tier, data source, and one-line rationale.

Two non-negotiable properties, both machine-checked by tests/test_leakage.py:
  1. every feature is computed from bars <= the bar it is reported at (prefix
     invariance under future-noise replacement);
  2. `source` declares whether a feature depends on close / volume / time, which is
     what the shift-by-1 test (plan A2) asserts per feature. Time-only features must
     be *invariant* to shifting close; everything else must move.

No realized-R, exit reason, or any forward-looking quantity appears in this module.
The feature matrix is built from the registry allow-list only — there is no code path
that feeds "all columns" to a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import (
    atr_wilder,
    ema,
    ffd,
    path_efficiency,
    realized_vol,
    rolling_kurt,
    rolling_pctile,
    rolling_skew,
    rolling_slope,
    rolling_zscore,
)

SourceCategory = str  # 'close' | 'volume' | 'time' | 'mixed'


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    tier: str
    source: SourceCategory
    window: int | None
    rationale: str
    scope: str = "bar"  # 'bar' (continuous series) or 'cross' (cross-level feature)
    transform_family: str = "continuous"  # 'continuous' | 'categorical' | 'binary'
    leak_expectation: str = "changes"  # 'changes' | 'invariant'
    alias_of: str | None = None  # duplicated information; excluded from the model pool


REGISTRY: dict[str, FeatureSpec] = {}


def _reg(name: str, tier: str, source: str, window: int | None, rationale: str,
         scope: str = "bar", transform_family: str = "continuous") -> str:
    # A2: only close/mixed features are expected to move when close is shifted.
    # Time-only features must be exactly invariant; pure-volume features are also
    # invariant to a close shift (declared, so the test cannot pass by accident).
    leak = "changes" if source in ("close", "mixed") else "invariant"
    if name in REGISTRY:
        raise ValueError(f"duplicate feature registration: {name}")
    REGISTRY[name] = FeatureSpec(name, tier, source, window, rationale, scope,
                                 transform_family, leak)
    return name


# ---------------------------------------------------------------------------
# Registry — Tier A (base), B (proposed/approved additions), FFD, CROSS, REGIME, INTER
# ---------------------------------------------------------------------------
for _k in (1, 3, 6, 12, 24):
    _reg(f"ret_{_k}", "A", "close", _k, f"log return over {_k}h: the local impulse producing the cross")

for _k in (24, 72):
    _reg(f"rv_{_k}", "A", "close", _k, f"realized vol over {_k}h: level of risk / TP reachability")
    _reg(f"eff_{_k}", "A", "close", _k, f"path efficiency over {_k}h: directional vs choppy approach to the cross")
    _reg(f"skew_{_k}", "D", "close", _k, f"return skew over {_k}h: burst asymmetry")
    _reg(f"kurt_{_k}", "D", "close", _k, f"return kurtosis over {_k}h: tail-shape of the recent path")

_reg("autocorr_1_24", "A", "close", 24, "lag-1 autocorrelation of 1h returns: persistence vs mean reversion")
_reg("atr_ratio_14_72", "A", "close", 72, "ATR14/ATR72: vol expansion or contraction at the cross")
_reg("sma_gap_72_atr", "A", "close", 72, "distance from the 72h SMA in ATR units: stretched or anchored")
_reg("sma_slope_24", "A", "close", 24, "72h SMA slope over 24h, in ATR/bar: trend direction and speed")
_reg("sma_slope_72", "A", "close", 72, "72h SMA slope over 72h, in ATR/bar: slower trend")
_reg("volume_z_24", "A", "volume", 24, "volume z-score: participation at the cross")
_reg("volume_z_72", "A", "volume", 72, "slower volume z-score: baseline participation")
for _c, _r in (("hour_sin", "cyclical UTC hour"), ("hour_cos", "cyclical UTC hour"),
               ("dow_sin", "cyclical day-of-week"), ("dow_cos", "cyclical day-of-week")):
    _reg(_c, "A", "time", None, f"{_r}: session seasonality without fake ordinality")
_reg("has_gap_24", "A", "time", 24, "any data gap > 1.5h in the last 24 bars: data-quality confounder")

for _k in (1, 3, 6, 12, 24):
    _reg(f"vret_{_k}", "B", "close", _k, f"return over {_k}h normalised by rolling 1h vol: regime-free momentum")

for _d in ("03", "04", "05"):
    _reg(f"ffd_{_d}", "F", "close", None,
         f"fractional difference of log close at d=0.{_d[1]}: stationary but memory-preserving")

_reg("ffd_selected", "F", "close", None,
     "fractional difference of log close at d selected by ADF on the TRAINING folds only "
     "(one d enters the model; all three are reported)", scope="cross")
_reg("cross_side", "X", "close", None, "long/short leg of the cross; the two legs are not symmetric",
     scope="cross", transform_family="categorical")
_reg("bars_since_prev_cross", "X", "close", None, "bars since the previous cross: fresh move vs whipsaw",
     scope="cross")
_reg("ema_spread_atr", "X", "close", None, "|EMA9-EMA21|/ATR at the cross: how decisive the cross is",
     scope="cross")
_reg("ema_spread_pctile_72", "X", "close", 72, "percentile of the EMA spread in its trailing 72h window",
     scope="cross")
_reg("ema9_slope_atr", "X", "close", 6, "EMA9 slope per bar in ATR units: fast-leg momentum",
     scope="cross")
_reg("ema21_slope_atr", "X", "close", 6, "EMA21 slope per bar in ATR units: slow-leg momentum",
     scope="cross")
_reg("dist_from_ema21_atr", "X", "close", None, "close minus EMA21 in ATR units: extension at the cross",
     scope="cross")
_reg("atr_pctile_72", "X", "close", 72, "ATR percentile in trailing 72h: vol regime at the cross",
     scope="cross")
_reg("volume_z_at_cross", "X", "volume", 24, "volume z-score on the cross bar", scope="cross")
_reg("volume_trend_12", "X", "volume", 12, "log(mean vol 6h / mean vol 12h): rising or fading participation",
     scope="cross")
_reg("signed_volume_z_24", "X", "mixed", 24,
     "volume z-score signed by return direction, averaged over 24h: buy/sell pressure", scope="cross")
_reg("clv_cross", "B", "close", None, "close location value on the cross bar: wick/rejection asymmetry",
     scope="cross")
_reg("consec_cross_24", "B", "close", 24, "number of crosses in the last 24h: whipsaw density", scope="cross")

_reg("eff_pctile_72", "R", "close", 72, "path-efficiency percentile: regime-relative directionality",
     scope="cross")
_reg("rv_pctile_72", "R", "close", 72, "realized-vol percentile: regime-relative risk", scope="cross")
_reg("dist_from_sma72_atr", "R", "close", 72,
     "distance from 72h SMA in ATR units: regime-relative stretch (identical to sma_gap_72_atr; "
     "kept in the registry for traceability, declared an alias and excluded from the pool)",
     scope="cross")
REGISTRY["dist_from_sma72_atr"] = FeatureSpec(
    "dist_from_sma72_atr", "R", "close", 72,
    "alias of sma_gap_72_atr (same formula); excluded from the model pool to avoid perfect collinearity",
    "cross", "continuous", "changes", alias_of="sma_gap_72_atr")

_reg("side_x_ema_spread", "I", "mixed", None, "cross_side x ema_spread_atr: decisiveness differs by leg",
     scope="cross")
_reg("side_x_eff_pctile", "I", "mixed", 72, "cross_side x efficiency percentile: trend-following edge by leg",
     scope="cross")
_reg("side_x_vol_pctile", "I", "mixed", 72, "cross_side x vol percentile: vol sensitivity by leg",
     scope="cross")
_reg("eff_x_vol_pctile", "I", "mixed", 72, "efficiency x vol percentile: clean trends need room to run",
     scope="cross")

BAR_FEATURES = tuple(k for k, v in REGISTRY.items() if v.scope == "bar")
CROSS_FEATURES = tuple(k for k, v in REGISTRY.items() if v.scope == "cross")
# candidate model pool = registered features minus declared aliases minus the
# unselected FFD variants (only `ffd_selected` enters; ffd_03/04/05 are diagnostics)
MODEL_POOL = tuple(
    k for k, v in REGISTRY.items() if v.alias_of is None and not k.startswith("ffd_0")
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _periods_per_year(timeframe: str = "1h") -> float:
    return {"1h": 24 * 365.25, "4h": 6 * 365.25, "1d": 365.25}.get(timeframe, 24 * 365.25)


def _gap_flag(ts: pd.Series, window: int, max_gap_hours: float) -> pd.Series:
    d = ts.diff().dt.total_seconds() / 3600.0
    is_gap = (d > max_gap_hours).astype(float).fillna(0.0)
    return is_gap.rolling(window, min_periods=1).max()


def build_bar_features(
    df: pd.DataFrame,
    rank_window: int = 72,
    max_gap_hours: float = 1.5,
    ffd_d_values: tuple[float, ...] = (0.3, 0.4, 0.5),
    timeframe: str = "1h",
) -> pd.DataFrame:
    """All bar-scope features, indexed by bar. Causality is structural: every rolling
    window ends at the current bar and EMAs are recursive with adjust=False."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    logc = np.log(close)
    logret = logc.diff()
    atr = pd.Series(atr_wilder(df, 14), index=df.index)
    out = pd.DataFrame(index=df.index)

    for k in (1, 3, 6, 12, 24):
        out[f"ret_{k}"] = logc.diff(k)
    for k in (24, 72):
        out[f"rv_{k}"] = realized_vol(logret, k)
        out[f"eff_{k}"] = path_efficiency(close, k)
        out[f"skew_{k}"] = rolling_skew(logret, k)
        out[f"kurt_{k}"] = rolling_kurt(logret, k)

    out["autocorr_1_24"] = logret.rolling(24, min_periods=24).corr(logret.shift(1))

    atr72 = pd.Series(atr_wilder(df, 72), index=df.index)
    out["atr_ratio_14_72"] = atr / atr72.replace(0.0, np.nan)

    sma72 = close.rolling(72, min_periods=72).mean()
    out["sma_gap_72_atr"] = (close - sma72) / atr.replace(0.0, np.nan)
    out["sma_slope_24"] = rolling_slope(sma72, 24) / atr.replace(0.0, np.nan)
    out["sma_slope_72"] = rolling_slope(sma72, 72) / atr.replace(0.0, np.nan)

    out["volume_z_24"] = rolling_zscore(vol, 24)
    out["volume_z_72"] = rolling_zscore(vol, 72)

    ts = pd.to_datetime(df["timestamp"])
    hod = ts.dt.hour.to_numpy(dtype=float)
    dow = ts.dt.day_of_week.to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["has_gap_24"] = _gap_flag(ts, 24, max_gap_hours)

    # volatility-normalised returns: ret_k / (rolling 1h vol * sqrt(k)). The vol window
    # is max(k, 24) so that k=1 (a single return has zero dispersion) stays defined.
    for k in (1, 3, 6, 12, 24):
        win = max(k, 24)
        denom = logret.rolling(win, min_periods=win).std(ddof=0) * np.sqrt(k)
        out[f"vret_{k}"] = out[f"ret_{k}"] / denom.replace(0.0, np.nan)

    for d in ffd_d_values:
        tag = f"{d:.1f}".replace(".", "")
        out[f"ffd_{tag}"] = ffd(logc, d)

    # keep ema/atr plumbing for cross-level features
    out["_ema_fast"] = ema(close, 9)
    out["_ema_slow"] = ema(close, 21)
    out["_atr"] = atr
    out["_close"] = close
    out["_high"] = high
    out["_low"] = low
    out["_volume"] = vol
    return out


def build_cross_matrix(
    df: pd.DataFrame,
    bar: pd.DataFrame,
    crosses: pd.DataFrame,
    rank_window: int = 72,
    ffd_d: float | None = None,
    ema_fast: int = 9,
    ema_slow: int = 21,
) -> pd.DataFrame:
    """Model feature pool at each cross. Rows align with `crosses`; every value is
    read from the cross bar or earlier."""
    idx = crosses["cross_idx"].to_numpy()
    at = bar.iloc[idx]
    close = bar["_close"].to_numpy()[idx]
    high = bar["_high"].to_numpy()[idx]
    low = bar["_low"].to_numpy()[idx]
    atr = bar["_atr"].to_numpy()[idx]
    ef = bar["_ema_fast"].to_numpy()[idx]
    es = bar["_ema_slow"].to_numpy()[idx]
    side = crosses["side"].to_numpy().astype(float)

    out = pd.DataFrame(index=crosses.index)
    # bar-scope features gathered at the cross (FFD kept to the selected d only)
    for name in BAR_FEATURES:
        if name.startswith("ffd_"):
            continue
        out[name] = at[name].to_numpy()
    out["ffd_selected"] = np.nan
    if ffd_d is not None:
        tag = f"ffd_{ffd_d:.1f}".replace(".", "")
        out["ffd_selected"] = at[tag].to_numpy()

    spread = np.abs(ef - es)
    out["cross_side"] = side
    if "bars_since_prev_cross" in crosses:
        out["bars_since_prev_cross"] = crosses["bars_since_prev_cross"].to_numpy(dtype=float)
    else:  # recompute causally from the cross indices if the caller did not supply it
        ci = crosses["cross_idx"].to_numpy(dtype=float)
        gaps = np.diff(ci, prepend=np.nan)
        out["bars_since_prev_cross"] = gaps
    out["ema_spread_atr"] = spread / np.where(atr > 0, atr, np.nan)
    # rolling percentile of the spread, evaluated causally at the cross bar
    spread_series = pd.Series(bar["_ema_fast"] - bar["_ema_slow"]).abs() / pd.Series(
        np.where(bar["_atr"] > 0, bar["_atr"], np.nan)
    )
    out["ema_spread_pctile_72"] = rolling_pctile(spread_series, rank_window).to_numpy()[idx]
    out["ema9_slope_atr"] = (
        (pd.Series(ef) - pd.Series(ef).shift(6)).to_numpy() / 6.0 / np.where(atr > 0, atr, np.nan)
    )
    out["ema21_slope_atr"] = (
        (pd.Series(es) - pd.Series(es).shift(6)).to_numpy() / 6.0 / np.where(atr > 0, atr, np.nan)
    )
    out["dist_from_ema21_atr"] = (close - es) / np.where(atr > 0, atr, np.nan)
    atr_series = pd.Series(bar["_atr"])
    out["atr_pctile_72"] = rolling_pctile(atr_series, rank_window).to_numpy()[idx]
    out["volume_z_at_cross"] = at["volume_z_24"].to_numpy()
    vol = bar["_volume"]
    out["volume_trend_12"] = np.log(
        (vol.rolling(6, min_periods=6).mean() / vol.rolling(12, min_periods=12).mean()).clip(lower=1e-12)
    ).to_numpy()[idx]
    logret = np.log(bar["_close"]).diff()
    sgn = np.sign(logret).fillna(0.0)
    signed = (sgn * bar["volume_z_24"]).rolling(24, min_periods=24).mean()
    out["signed_volume_z_24"] = signed.to_numpy()[idx]
    rng = np.where((high - low) > 0, high - low, np.nan)
    out["clv_cross"] = (close - low) / rng
    cross_flag = pd.Series(np.isin(np.arange(len(df)), idx).astype(float))
    out["consec_cross_24"] = cross_flag.rolling(24, min_periods=1).sum().to_numpy()[idx]

    # regime-relative context
    out["eff_pctile_72"] = rolling_pctile(bar["eff_24"], rank_window).to_numpy()[idx]
    out["rv_pctile_72"] = rolling_pctile(bar["rv_24"], rank_window).to_numpy()[idx]
    out["dist_from_sma72_atr"] = out["sma_gap_72_atr"]  # declared alias, dropped from the pool

    # explicit interactions
    out["side_x_ema_spread"] = out["cross_side"] * out["ema_spread_atr"]
    out["side_x_eff_pctile"] = out["cross_side"] * out["eff_pctile_72"]
    out["side_x_vol_pctile"] = out["cross_side"] * out["rv_pctile_72"]
    out["eff_x_vol_pctile"] = out["eff_pctile_72"] * out["rv_pctile_72"]
    return out


def rank_transform(X: pd.DataFrame, registry: dict[str, FeatureSpec] | None = None,
                   rank_window: int = 72, bar: pd.DataFrame | None = None,
                   crosses: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trailing-window percentile rank of every continuous feature, evaluated causally.

    Used for the linear models (M1-M3) per the plan's tier policy: raw features for
    trees, rank-transformed in place for linear models, so `p` stays comparable across
    the ladder. `bar`/`crosses` must be supplied for the cross-scope features to be
    ranked on their real trailing windows; bar-scope features are passed through when
    their trailing values are unavailable (flagged by the caller).
    """
    reg = registry or REGISTRY
    out = X.copy()
    for name in X.columns:
        spec = reg.get(name)
        if spec is None or spec.transform_family != "continuous":
            continue
        if bar is None:
            out[name] = rolling_pctile(X[name], rank_window)
            continue
        out[name] = _cross_rank(name, X, bar, crosses, rank_window)
    return out


def _cross_rank(name: str, X: pd.DataFrame, bar: pd.DataFrame, crosses: pd.DataFrame,
                rank_window: int) -> pd.Series:
    """Percentile rank of a cross-level value within its trailing window on the bar
    series where that is well defined; otherwise rank within the trailing window of
    cross observations (still causal: only past crosses)."""
    if crosses is None:
        return rolling_pctile(X[name], rank_window)
    idx = crosses["cross_idx"].to_numpy()
    vals = X[name].to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if name in BAR_FEATURES and name in bar.columns:
        full = rolling_pctile(bar[name], rank_window).to_numpy()
        return pd.Series(full[idx], index=X.index)
    # cross-level: trailing window over past crosses (causal, no forward references)
    for k in range(len(vals)):
        lo = max(0, k - rank_window + 1)
        window = vals[lo : k + 1]
        window = window[np.isfinite(window)]
        if len(window) >= 5 and np.isfinite(vals[k]):
            out[k] = float((window <= vals[k]).mean())
    return pd.Series(out, index=X.index)


def select_ffd_d(
    log_close: pd.Series,
    train_end: int | None,
    d_values: tuple[float, ...] = (0.3, 0.4, 0.5),
    p_threshold: float = 0.05,
) -> dict:
    """Choose the FFD order for ONE training block (never using validation or hold-out).

    Rule (pre-registered): the smallest d whose ADF p-value on the TRAINING bars is
    below `p_threshold`, i.e. the most memory-preserving stationary differencing. If
    none reach the threshold, the most stationary d is used and flagged.
    """
    from statsmodels.tsa.stattools import adfuller

    x = log_close.iloc[: train_end] if train_end is not None else log_close
    res: dict = {"p_values": {}, "stationary": {}, "chosen_d": None, "rule": None,
                 "n_train_bars": int(len(x))}
    p_log = np.nan
    try:
        p_log = float(adfuller(x.dropna().to_numpy(), maxlag=10, autolag="AIC")[1])
    except Exception:
        pass
    res["p_values"]["log_price"] = p_log
    for d in d_values:
        y = ffd(x, d).dropna().to_numpy()
        p = np.nan
        if len(y) > 60:
            try:
                p = float(adfuller(y, maxlag=10, autolag="AIC")[1])
            except Exception:
                p = np.nan
        res["p_values"][f"d={d:.1f}"] = p
        res["stationary"][f"d={d:.1f}"] = bool(np.isfinite(p) and p < p_threshold)
    cands = [d for d in sorted(d_values) if res["stationary"][f"d={d:.1f}"]]
    if cands:
        res["chosen_d"] = float(cands[0])
        res["rule"] = "smallest d with ADF p < threshold on training bars"
    else:
        finite = {d: res["p_values"][f"d={d:.1f}"] for d in d_values}
        finite = {k: v for k, v in finite.items() if np.isfinite(v)}
        if finite:
            res["chosen_d"] = float(min(finite, key=finite.get))
            res["rule"] = "NO d reached the ADF threshold; least non-stationary d used (flagged)"
        else:
            res["chosen_d"] = float(d_values[-1])
            res["rule"] = "ADF failed for every d; defaulted to the largest d (flagged)"
    return res


def feature_count_report(selected: list[str]) -> dict:
    tiers: dict[str, int] = {}
    for name in selected:
        spec = REGISTRY.get(name)
        tier = spec.tier if spec else "?"
        tiers[tier] = tiers.get(tier, 0) + 1
    return {"n_features": len(selected), "by_tier": tiers}
