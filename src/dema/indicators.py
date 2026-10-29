"""Causal indicator primitives.

Every function here reads only current and past bars. That property is enforced
mechanically by tests/test_leakage.py (prefix invariance under future noise), not
by convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Recursive EMA. adjust=False is mandatory: adjust=True normalises over the
    full sample and therefore leaks future bars into early values."""
    return series.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()


def true_range(df: pd.DataFrame) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return tr.to_numpy(dtype=float)


def atr_wilder(df: pd.DataFrame, window: int = 14) -> np.ndarray:
    """Wilder ATR: seeded with the SMA of the first `window` true ranges, then
    recursive. Causal by construction (value at i uses bars <= i)."""
    tr = true_range(df)
    n = len(tr)
    out = np.full(n, np.nan)
    if n < window:
        return out
    out[window - 1] = np.nanmean(tr[:window])
    alpha = 1.0 / float(window)
    for i in range(window, n):
        out[i] = out[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return out


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=window).mean()
    s = x.rolling(window, min_periods=window).std(ddof=0)
    return (x - m) / s.replace(0.0, np.nan)


def rolling_pctile(x: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the current value within its trailing window (causal)."""
    return x.rolling(window, min_periods=window).apply(
        lambda a: float((a <= a[-1]).mean()), raw=True
    )


def rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of x over the trailing window, per bar."""
    idx = np.arange(window, dtype=float)
    idx -= idx.mean()
    denom = float((idx**2).sum())

    def _slope(a: np.ndarray) -> float:
        return float((idx * (a - a.mean())).sum() / denom)

    return x.rolling(window, min_periods=window).apply(_slope, raw=True)


def path_efficiency(close: pd.Series, window: int) -> pd.Series:
    """|net move| / sum|bar moves| over the trailing window. 1.0 = perfectly
    directional, ~0 = pure chop."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0.0, np.nan)


def realized_vol(logret: pd.Series, window: int) -> pd.Series:
    return logret.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(window)


def rolling_skew(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).skew()


def rolling_kurt(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window).kurt()


def ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Fractional-difference weights ordered by LAG: w[j] is the weight on x[t-j].

    w[0] = 1, w[k] = -w[k-1] * (d - k + 1) / k, truncated once |w| <= threshold.
    """
    w, k = [1.0], 1
    while abs(w[-1]) > threshold:
        w.append(-w[-1] * (d - k + 1) / k)
        k += 1
    return np.array(w[:-1], dtype=float)


def ffd_length(d: float, threshold: float = 1e-5) -> int:
    return int(len(ffd_weights(d, threshold)))


def ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Causal fractional differentiation: y[t] = sum_j w[j] * x[t-j].

    Implemented as a causal convolution, so y[t] depends only on bars <= t. The first
    len(w)-1 values are NaN (startup). NOTE the data cost: at d=0.4 and |w|<1e-5 the
    window is ~1.4k bars, which is subtracted from usable history in the audit.
    """
    w = ffd_weights(d, threshold)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(w) > len(x):
        return pd.Series(out, index=series.index)
    conv = np.convolve(x, w, mode="full")[: len(x)]
    out[len(w) - 1 :] = conv[len(w) - 1 :]
    return pd.Series(out, index=series.index)


def sign_changes(x: np.ndarray) -> np.ndarray:
    s = np.sign(x)
    cross = np.zeros(len(s), dtype=bool)
    cross[1:] = (s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0)
    return cross
