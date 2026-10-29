"""OHLCV loading, integrity auditing, and the (clearly labelled) synthetic fallback."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
SYNTHETIC_BANNER = "SYNTHETIC - not tradeable"


class DataIntegrityError(RuntimeError):
    pass


@dataclass
class DataAudit:
    source: str
    synthetic: bool
    n_rows: int = 0
    n_rows_dropped: int = 0
    start: str | None = None
    end: str | None = None
    median_bar_seconds: float | None = None
    n_gaps: int = 0
    longest_gap_hours: float | None = None
    duplicates: int = 0
    unsorted: int = 0
    negative_or_zero_prices: int = 0
    negative_volume: int = 0
    nan_rows: int = 0
    bad_ohlc: int = 0
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """Stable hash of the (sorted, normalised) bar frame that actually feeds the model."""
    payload = df[list(REQUIRED_COLUMNS)].to_csv(index=False, float_format="%.10g").encode()
    return hashlib.sha256(payload).hexdigest()


def load_ohlcv(
    path: str | Path,
    timezone: str = "UTC",
    max_gap_hours: float = 1.5,
) -> tuple[pd.DataFrame, DataAudit]:
    """Load a real CSV with columns timestamp,open,high,low,close,volume."""
    path = Path(path)
    if not path.exists():
        raise DataIntegrityError(f"CSV not found: {path}")
    raw = pd.read_csv(path)
    audit = DataAudit(source=f"file:{path}", synthetic=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataIntegrityError(f"{path}: missing required columns {missing}")
    df = _normalise(raw, audit, timezone, max_gap_hours)
    return df, audit


def _normalise(raw: pd.DataFrame, audit: DataAudit, timezone: str, max_gap_hours: float) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)

    # timestamps -------------------------------------------------------------
    ts_raw = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts_raw):
        unit = "ms" if float(np.nanmax(ts_raw.to_numpy(dtype=float))) > 1e11 else "s"
        ts = pd.to_datetime(ts_raw, unit=unit, utc=True, errors="coerce")
        audit.issues.append(f"numeric timestamps interpreted as epoch {unit}")
    else:
        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    n_bad_ts = int(ts.isna().sum())
    if n_bad_ts:
        audit.issues.append(f"{n_bad_ts} unparseable timestamps dropped")
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]
    df["timestamp"] = ts.astype("datetime64[ns, UTC]").to_numpy()

    # duplicates -------------------------------------------------------------
    dup = df["timestamp"].duplicated(keep="last")
    audit.duplicates = int(dup.sum())
    if audit.duplicates:
        audit.issues.append(f"{audit.duplicates} duplicate timestamps dropped (kept last)")
        df = df.loc[~dup].copy()

    # order ------------------------------------------------------------------
    order = df["timestamp"].argsort(kind="mergesort")
    audit.unsorted = int((order.to_numpy() != np.arange(len(df))).sum())
    if audit.unsorted:
        audit.issues.append(f"{audit.unsorted} rows were out of chronological order (sorted)")
    df = df.iloc[order].reset_index(drop=True)

    # numeric coercion -------------------------------------------------------
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nan_rows = df[list(REQUIRED_COLUMNS[1:])].isna().any(axis=1)
    audit.nan_rows = int(nan_rows.sum())
    if audit.nan_rows:
        audit.issues.append(f"{audit.nan_rows} rows with NaN/invalid OHLCV dropped")

    bad_nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    audit.negative_or_zero_prices = int(bad_nonpos.sum())
    if audit.negative_or_zero_prices:
        audit.issues.append(f"{audit.negative_or_zero_prices} rows with <=0 price dropped")

    neg_vol = df["volume"] < 0
    audit.negative_volume = int(neg_vol.sum())
    if audit.negative_volume:
        audit.issues.append(f"{audit.negative_volume} rows with negative volume set to NaN->dropped")

    bad_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    audit.bad_ohlc = int(bad_ohlc.sum())
    if audit.bad_ohlc:
        audit.issues.append(f"{audit.bad_ohlc} rows violating high>=max(o,c), low<=min(o,c), high>=low dropped")

    keep = ~(nan_rows | bad_nonpos | neg_vol | bad_ohlc)
    df = df.loc[keep].reset_index(drop=True)
    audit.n_rows_dropped = int(n0 - len(df))
    audit.n_rows = int(len(df))
    if audit.n_rows < 500:
        raise DataIntegrityError(f"only {audit.n_rows} usable bars after cleaning; need >= 500")

    # bar grid + gap census --------------------------------------------------
    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    median_s = float(deltas.median()) if len(deltas) else np.nan
    audit.median_bar_seconds = median_s
    if median_s and not (0.9 * 3600 <= median_s <= 1.1 * 3600):
        audit.issues.append(
            f"median bar spacing is {median_s/3600:.2f}h, not 1h — this pipeline assumes 1h bars"
        )
    gaps = deltas[deltas > max_gap_hours * 3600]
    audit.n_gaps = int(len(gaps))
    audit.longest_gap_hours = round(float(gaps.max() / 3600.0), 2) if len(gaps) else 0.0
    if audit.n_gaps:
        audit.issues.append(
            f"{audit.n_gaps} bar gaps > {max_gap_hours}h (longest {audit.longest_gap_hours}h); "
            "NOT forward-filled — features spanning a gap are flagged has_gap_24"
        )

    audit.start = str(df["timestamp"].iloc[0])
    audit.end = str(df["timestamp"].iloc[-1])
    return df[list(REQUIRED_COLUMNS)].reset_index(drop=True)


def synthetic_ohlcv(
    n_bars: int = 26280,
    start: str = "2022-01-01T00:00:00Z",
    seed: int = 7,
    base_vol: float = 0.006,
    vol_persistence: float = 0.97,
    vol_of_vol: float = 0.10,
    t_df: float = 4.0,
    intraday_seasonality: bool = True,
    s0: float = 30000.0,
) -> tuple[pd.DataFrame, DataAudit]:
    """Deterministic synthetic 1h series: fat-tailed returns with AR(1) log-vol and
    intraday seasonality. Watermarked SYNTHETIC everywhere it is used.

    This is a *test fixture*, never evidence. Any metric computed on it is prefixed
    SYNTHETIC and excluded from the report's verdict.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=pd.Timestamp(start), periods=n_bars, freq="h", tz="UTC")

    log_vol = np.zeros(n_bars)
    log_vol[0] = np.log(base_vol)
    for i in range(1, n_bars):
        log_vol[i] = (
            np.log(base_vol)
            + vol_persistence * (log_vol[i - 1] - np.log(base_vol))
            + vol_of_vol * rng.standard_normal()
        )
    vol = np.exp(log_vol)

    season = np.ones(n_bars)
    if intraday_seasonality:
        hours = ts.hour.to_numpy()
        season = 0.75 + 0.55 * np.exp(-((hours - 15.0) ** 2) / (2 * 6.0**2))  # peak ~15:00 UTC

    # mild trend/mean-reversion mixture so the EMA crossover has something to trade
    drift = 0.02 * vol * rng.standard_normal(n_bars)
    shocks = rng.standard_t(t_df, n_bars) / np.sqrt(t_df / (t_df - 2.0))
    ret = drift + vol * season * shocks

    close = s0 * np.exp(np.cumsum(ret))
    # Intrabar geometry: open near previous close with a small gap, high/low drawn
    # around the open-close range so that OHLC identities hold exactly.
    gap = rng.standard_normal(n_bars) * vol * 0.15
    open_ = np.empty(n_bars)
    open_[0] = s0
    open_[1:] = close[:-1] * np.exp(gap[1:])
    wick_up = np.abs(rng.standard_normal(n_bars)) * vol * season * 0.7
    wick_dn = np.abs(rng.standard_normal(n_bars)) * vol * season * 0.7
    high = np.maximum(open_, close) * np.exp(wick_up)
    low = np.minimum(open_, close) * np.exp(-wick_dn)
    volume = (1e3 * np.exp(1.2 * rng.standard_normal(n_bars)) * (1.0 + 40.0 * np.abs(ret))).round()

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    audit = DataAudit(
        source=f"synthetic(seed={seed}, n_bars={n_bars})",
        synthetic=True,
        n_rows=n_bars,
        start=str(ts[0]),
        end=str(ts[-1]),
        median_bar_seconds=3600.0,
        n_gaps=0,
        longest_gap_hours=0.0,
        issues=[SYNTHETIC_BANNER],
    )
    return df, audit


def get_data(cfg: dict, smoke_test: bool, data_path: str | None = None) -> tuple[pd.DataFrame, DataAudit]:
    """Resolve the data source. Real CSV wins; synthetic only when explicitly allowed."""
    path = data_path or cfg["data"].get("path")
    if path and not smoke_test:
        return load_ohlcv(
            path,
            timezone=cfg["data"].get("timezone", "UTC"),
            max_gap_hours=float(cfg["data"].get("max_gap_hours_report", 1.5)),
        )
    if path and smoke_test:
        # --smoke-test with a real path still loads synthetic: never silently train on real data
        pass
    s = cfg["synthetic"]
    return synthetic_ohlcv(
        n_bars=int(s["n_bars"]),
        start=str(s["start"]),
        seed=int(s["seed"]),
        base_vol=float(s["base_vol"]),
        vol_persistence=float(s["vol_persistence"]),
        vol_of_vol=float(s["vol_of_vol"]),
        t_df=float(s["t_df"]),
        intraday_seasonality=bool(s["intraday_seasonality"]),
    )
