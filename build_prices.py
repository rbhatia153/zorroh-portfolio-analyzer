
# scripts/build_prices.py
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

ETF_UNIVERSE = [
    "SPY","QQQ","IWM","EFA","EEM","AGG","BND","LQD","HYG","TLT",
    "IEF","GLD","SLV","DBC","VNQ","XLB","XLE","XLF","XLI","XLK",
    "XLP","XLRE","XLU","XLV","XLY",
]

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            df = df["Close"].copy()
        else:
            for field in ("Adj Close", "Close"):
                if field in df.columns.levels[0]:
                    df = df[field].copy()
                    break
        df.columns.name = None
        return df
    new_cols = []
    for c in df.columns:
        name = str(c).replace(" Adj Close","").replace(" Close","").strip()
        new_cols.append(name)
    df = df.copy()
    df.columns = new_cols
    return df

def main():
    start = dt.date(2010, 1, 1)
    end   = dt.date.today()
    print(f"Downloading prices {start} → {end} for {len(ETF_UNIVERSE)} ETFs…")
    raw = yf.download(ETF_UNIVERSE, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise SystemExit("No data returned from yfinance")

    prices = _sanitize_columns(raw)
    prices = prices.sort_index()
    Path("data").mkdir(parents=True, exist_ok=True)
    out = Path("data") / "prices.parquet"
    prices.to_parquet(out)
    print(f"✅ Wrote {out} with shape {prices.shape}")

if __name__ == "__main__":
    main()