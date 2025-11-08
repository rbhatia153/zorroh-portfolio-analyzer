from __future__ import annotations
import pandas as pd

def compute_simple_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' in DataFrame")
    return df[col].pct_change().dropna()
