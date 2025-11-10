"""
📊 Zorroh Portfolio Analyzer — Mobile-optimized with ETF Performance Tab
ETF portfolio construction, analysis, and comprehensive ETF reference tables.

Run: streamlit run portfolio_app.py
"""

import datetime as dt
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

# ----------------------
# Responsive Helpers (no-JS, URL param persists)
# ----------------------
def _qp_get() -> dict:
    """Return query params as a plain dict across Streamlit versions."""
    try:
        qp = st.query_params
        if hasattr(qp, 'to_dict'):
            return qp.to_dict()
        return dict(qp)
    except Exception:
        pass
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}


def _qp_set(**kwargs):
    """Set query params across Streamlit versions."""
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v
        return True
    except Exception:
        pass
    try:
        st.experimental_set_query_params(**kwargs)
        return True
    except Exception:
        return False


def get_view_param() -> str | None:
    """Read ?view= from URL. Returns 'mobile', 'desktop', or None."""
    qp = _qp_get()
    v = qp.get("view")
    if isinstance(v, list):
        v = v[0] if v else None
    return v


def set_view_param(view: str):
    """Set ?view= in URL without immediate rerun."""
    _qp_set(view=view)
    st.session_state["_view_mode"] = view


def get_view_mode_with_default(default: str = "desktop") -> str:
    """
    Return 'mobile' or 'desktop'. Initialize from URL or session state.
    No automatic reruns - caller controls when to rerun.
    """
    if "_view_mode" in st.session_state:
        mode = st.session_state["_view_mode"]
        if mode in ("mobile", "desktop"):
            return mode
    
    v = get_view_param()
    if v in ("mobile", "desktop"):
        st.session_state["_view_mode"] = v
        return v
    
    st.session_state["_view_mode"] = default
    _qp_set(view=default)
    return default


# ----------------------
# Configuration
# ----------------------
VIEW_MODE = get_view_mode_with_default(default="desktop")
IS_MOBILE = (VIEW_MODE == "mobile")

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 100%;
}
[data-testid="stMetricValue"] { font-size: 1.1rem; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
table { width: 100%; }

.view-toggle-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 1rem 0;
    gap: 0.5rem;
}

@media (max-width: 1024px) {
  .block-container {
    padding-left: 0.6rem;
    padding-right: 0.6rem;
    padding-top: 0.5rem;
  }
  [data-testid="stMetricValue"] { font-size: 1rem; }
  [data-testid="stMetricDelta"] { font-size: 0.85rem; }
  .stPlotlyChart { height: auto !important; }
  table { font-size: 0.85rem; }
  button { min-height: 44px; }
  [data-testid="stSelectbox"] { font-size: 0.9rem; }
}

@media (min-width: 768px) and (max-width: 1024px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}

@media (min-width: 1025px) {
  .block-container {
    padding-left: 2rem;
    padding-right: 2rem;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "BND", "LQD", "HYG", "TLT",
    "IEF", "GLD", "SLV", "DBC", "VNQ", "XLB", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
]

REBALANCE_OPTIONS = {
    "Buy & Hold (No Rebalance)": None,
    "Monthly": "M",
    "Quarterly": "Q",
    "Annual": "Y",
}

DEFAULT_HOLDINGS = ["SPY", "EFA", "AGG", "VNQ", "GLD"]
DEFAULT_WEIGHTS = [40, 20, 25, 10, 5]
DEFAULT_BENCHMARK = "SPY"

# ----------------------
# Robust price loading
# ----------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_prices(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Robust loader with parquet cache support."""
    if not tickers:
        return pd.DataFrame()

    if isinstance(start, dt.date) and isinstance(end, dt.date) and start > end:
        start, end = end, start

    want = sorted(set(tickers))
    cached = _try_read_cached_prices()

    pieces = []
    covered = set()

    if cached is not None:
        c = cached.copy()
        c = c.loc[(c.index >= pd.Timestamp(start)) & (c.index <= pd.Timestamp(end))]
        c = _sanitize_columns(c)
        have = [t for t in want if t in c.columns]
        if have:
            pieces.append(c[have])
            covered.update(have)

    missing = [t for t in want if t not in covered]
    if missing:
        try:
            df = yf.download(missing, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            fetched = _yf_to_wide(df, missing)
            if not fetched.empty:
                pieces.append(fetched)
        except Exception as e:
            st.warning(f"Live fetch failed for {missing}: {e}")

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, axis=1)
    out = _sanitize_columns(out)
    out = out.reindex(columns=want)
    out = out.dropna(how="all")
    return out


def _try_read_cached_prices() -> pd.DataFrame | None:
    """Look for data/prices.parquet built by nightly job."""
    p = Path("data") / "prices.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to plain tickers."""
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
        if isinstance(c, tuple):
            c = c[-1]
        name = str(c).replace(" Adj Close", "").replace(" Close", "").strip()
        new_cols.append(name)
    df = df.copy()
    df.columns = new_cols
    return df


def _yf_to_wide(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Convert yfinance download to 2D wide frame of Close prices."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            w = raw["Close"].copy()
        else:
            w = raw.xs(raw.columns.levels[0][0], axis=1, level=0)
    else:
        w = raw[["Close"]].copy()
        w.columns = [tickers[0]]
    w = _sanitize_columns(w)
    if not isinstance(w.index, pd.DatetimeIndex):
        w.index = pd.to_datetime(w.index)
    return w.sort_index()


# ----------------------
# Rebalancing & stats
# ----------------------
def get_rebalance_dates(prices: pd.DataFrame, freq: str) -> List[pd.Timestamp]:
    """First available trading day ON/AFTER each period start."""
    if freq is None:
        return []
    idx = prices.index
    if len(idx) == 0:
        return []
    if freq == "M":
        candidates = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="MS")
    elif freq == "Q":
        candidates = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="QS")
    elif freq == "Y":
        candidates = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="YS")
    else:
        return []
    aligned = []
    if len(idx):
        aligned.append(idx[0])
    for d in candidates:
        pos = idx.searchsorted(d)
        if pos < len(idx):
            aligned.append(idx[pos])
    return sorted(set(aligned))


def compute_equity_curve(prices: pd.DataFrame, weights: Dict[str, float], rebal_freq: str) -> pd.Series:
    """Portfolio equity curve with optional rebalancing."""
    prices = prices.dropna(how="all")
    returns = prices.pct_change().fillna(0.0)
    rebal_dates = set(get_rebalance_dates(prices, rebal_freq))
    pv = pd.Series(index=prices.index, dtype=float)
    if len(prices.index) == 0:
        return pv
    pv.iloc[0] = 1.0
    current = weights.copy()
    for i in range(1, len(prices)):
        d = prices.index[i]
        if rebal_freq is not None and d in rebal_dates:
            current = weights.copy()
        port_ret = sum(current.get(t, 0) * returns.iloc[i].get(t, 0.0) for t in weights)
        pv.iloc[i] = pv.iloc[i - 1] * (1 + port_ret)
        if rebal_freq is None or d not in rebal_dates:
            tot = sum(current.get(t, 0) * (1 + returns.iloc[i].get(t, 0.0)) for t in weights)
            if tot > 0:
                current = {t: current.get(t, 0) * (1 + returns.iloc[i].get(t, 0.0)) / tot for t in weights}
    return pv


def compute_weight_path(prices: pd.DataFrame, weights: Dict[str, float], rebal_freq: str) -> pd.DataFrame:
    """Actual weights each day under chosen rebalancing."""
    tickers = list(weights.keys())
    px = prices[tickers].dropna(how="all").copy()
    rets = px.pct_change().fillna(0.0)
    rebal_dates = set(get_rebalance_dates(px, rebal_freq))
    if len(px.index):
        rebal_dates.add(px.index[0])
    total_value = 1.0
    pos_values = {t: total_value * weights[t] for t in tickers}
    rows = []
    for d in px.index:
        for t in tickers:
            r = rets.at[d, t] if t in rets.columns else 0.0
            pos_values[t] *= (1.0 + (r if pd.notna(r) else 0.0))
        if rebal_freq is not None and d in rebal_dates:
            total_value = sum(pos_values.values())
            for t in tickers:
                pos_values[t] = total_value * weights[t]
        total_value = sum(pos_values.values())
        rows.append({t: (pos_values[t] / total_value if total_value > 0 else weights[t]) for t in tickers})
    return pd.DataFrame(rows, index=px.index, columns=tickers)


def max_drawdown(series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    cum_max = series.cummax()
    dd = (series - cum_max) / cum_max
    if dd.empty or dd.isna().all():
        return 0.0, (series.index[0] if len(series) else pd.NaT), (series.index[0] if len(series) else pd.NaT)
    trough = dd.idxmin()
    peak = series.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def drawdown_series(series: pd.Series) -> pd.Series:
    return (series - series.cummax()) / series.cummax()


def perf_stats(equity: pd.Series) -> Dict[str, float]:
    r = equity.pct_change().dropna()
    if len(r) < 30:
        return {"CAGR": 0.0, "Ann. Vol": 0.0, "Sharpe": 0.0, "Max DD": 0.0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = r.std() * np.sqrt(252)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    mdd, _, _ = max_drawdown(equity)
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe, "Max DD": mdd}


def tracking_error(port_r: pd.Series, bench_r: pd.Series) -> float:
    a = (port_r - bench_r).dropna()
    return a.std() * np.sqrt(252)


def information_ratio(port_r: pd.Series, bench_r: pd.Series) -> float:
    a = (port_r - bench_r).dropna()
    te = a.std() * np.sqrt(252)
    return ((a.mean() * 252) / te) if te > 0 else 0.0


def beta_corr(port_r: pd.Series, bench_r: pd.Series) -> Tuple[float, float]:
    df = pd.DataFrame({"port": port_r, "bench": bench_r}).dropna()
    if len(df) < 2:
        return 0.0, 0.0
    cov = df.cov()
    beta = cov.loc["port", "bench"] / cov.loc["bench", "bench"] if cov.loc["bench", "bench"] > 0 else 0.0
    return beta, df["port"].corr(df["bench"])


# ----------------------
# ETF Performance Helpers
# ----------------------
def period_return(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    """Total return between two dates for each column (ETFs). Align to nearest available dates."""
    ret = pd.Series(index=df.columns, dtype=float)
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            ret[col] = np.nan
            continue
        
        start_idx = series.index.searchsorted(start_dt)
        end_idx = series.index.searchsorted(end_dt)
        
        if start_idx >= len(series):
            start_idx = len(series) - 1
        if end_idx >= len(series):
            end_idx = len(series) - 1
        
        if start_idx == end_idx or start_idx < 0 or end_idx < 0:
            ret[col] = np.nan
        else:
            ret[col] = (series.iloc[end_idx] / series.iloc[start_idx]) - 1
    
    return ret


def trailing_cagr(df: pd.DataFrame, years: int, end_dt: pd.Timestamp) -> pd.Series:
    """CAGR over trailing N years, aligning start to nearest previous date."""
    start_dt = end_dt - relativedelta(years=years)
    ret = pd.Series(index=df.columns, dtype=float)
    
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            ret[col] = np.nan
            continue
        
        start_idx = series.index.searchsorted(start_dt)
        end_idx = series.index.searchsorted(end_dt)
        
        if start_idx >= len(series):
            start_idx = len(series) - 1
        if end_idx >= len(series):
            end_idx = len(series) - 1
        
        if start_idx >= end_idx or start_idx < 0:
            ret[col] = np.nan
            continue
        
        start_val = series.iloc[start_idx]
        end_val = series.iloc[end_idx]
        actual_days = (series.index[end_idx] - series.index[start_idx]).days
        
        if actual_days < years * 300:
            ret[col] = np.nan
        else:
            actual_years = actual_days / 365.25
            ret[col] = (end_val / start_val) ** (1 / actual_years) - 1
    
    return ret


def calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide table: rows=tickers, cols=YYYY, values=total return per calendar year + YTD."""
    results = []
    
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        row = {"Ticker": col}
        years = sorted(series.index.year.unique())
        
        for year in years:
            year_data = series[series.index.year == year]
            if len(year_data) >= 2:
                row[str(year)] = (year_data.iloc[-1] / year_data.iloc[0]) - 1
        
        current_year = series.index[-1].year
        ytd_data = series[series.index.year == current_year]
        if len(ytd_data) >= 2:
            row["YTD"] = (ytd_data.iloc[-1] / ytd_data.iloc[0]) - 1
        
        results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    df_out = pd.DataFrame(results)
    df_out = df_out.set_index("Ticker")
    return df_out


def long_term_risk_metrics(df: pd.DataFrame, bench: str = "SPY") -> pd.DataFrame:
    """Compute Ann Vol, Sharpe, Sortino, MaxDD, %DownMonths, Worst/Best Month, Beta, Corr vs SPY."""
    results = []
    
    bench_rets = None
    if bench in df.columns:
        bench_series = df[bench].dropna()
        if len(bench_series) > 0:
            bench_rets = bench_series.pct_change().dropna()
    
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 30:
            continue
        
        daily_rets = series.pct_change().dropna()
        monthly_series = series.resample("M").last()
        monthly_rets = monthly_series.pct_change().dropna()
        
        row = {"Ticker": col}
        
        row["Ann. Vol"] = daily_rets.std() * np.sqrt(252)
        row["Sharpe"] = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
        
        downside = daily_rets[daily_rets < 0]
        if len(downside) > 0:
            downside_std = downside.std() * np.sqrt(252)
            row["Sortino"] = (daily_rets.mean() * 252) / downside_std if downside_std > 0 else 0
        else:
            row["Sortino"] = 0
        
        equity = series / series.iloc[0]
        dd = (equity / equity.cummax()) - 1
        row["Max DD"] = dd.min()
        
        row["% Down Months"] = (monthly_rets < 0).sum() / len(monthly_rets) if len(monthly_rets) > 0 else np.nan
        row["Worst Month"] = monthly_rets.min() if len(monthly_rets) > 0 else np.nan
        row["Best Month"] = monthly_rets.max() if len(monthly_rets) > 0 else np.nan
        
        if bench_rets is not None and col != bench:
            aligned = pd.DataFrame({"asset": daily_rets, "bench": bench_rets}).dropna()
            if len(aligned) >= 30:
                cov_mat = aligned.cov()
                row["Beta"] = cov_mat.loc["asset", "bench"] / cov_mat.loc["bench", "bench"] if cov_mat.loc["bench", "bench"] > 0 else np.nan
                row["Corr"] = aligned["asset"].corr(aligned["bench"])
            else:
                row["Beta"] = np.nan
                row["Corr"] = np.nan
        else:
            row["Beta"] = 1.0 if col == bench else np.nan
            row["Corr"] = 1.0 if col == bench else np.nan
        
        results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_universe_prices(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Load all ETF universe prices for performance tables."""
    return load_prices(ETF_UNIVERSE, start, end)


# ----------------------
# UI
# ----------------------
st.title("📊 Zorroh Portfolio Analyzer")
st.caption("Build and analyze diversified ETF portfolios vs a benchmark")

# Create main tabs
tabs = st.tabs(["📊 Analyzer", "📈 ETF Performance"])

# ===========================
# TAB 1: ANALYZER
# ===========================
with tabs[0]:
    col_left, col_toggle, col_right = st.columns([1, 2, 1])
    with col_toggle:
        st.markdown('<div class="view-toggle-container">', unsafe_allow_html=True)
        
        view_choice = st.radio(
            "View Mode",
            options=["📱 Mobile", "🖥️ Desktop"],
            index=0 if IS_MOBILE else 1,
            horizontal=True,
            key="view_toggle",
            label_visibility="collapsed",
        )
        
        new_view = "mobile" if "Mobile" in view_choice else "desktop"
        if new_view != VIEW_MODE:
            set_view_param(new_view)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    if Path("data/prices.parquet").exists():
        st.caption("⚡ Using precomputed data from `data/prices.parquet`")
    else:
        st.caption("⏳ No precomputed parquet found; will fetch live data")

    def H(desktop: int, mobile: int) -> int:
        return mobile if IS_MOBILE else desktop

    st.sidebar.header("Portfolio Setup")
    st.sidebar.subheader("Holdings (max 5)")
    holdings, weights_input = [], []
    for i in range(5):
        c1, c2 = st.sidebar.columns([3, 2])
        default_t = DEFAULT_HOLDINGS[i] if i < len(DEFAULT_HOLDINGS) else None
        default_w = DEFAULT_WEIGHTS[i] if i < len(DEFAULT_WEIGHTS) else 0
        ticker = c1.selectbox(
            f"ETF {i + 1}",
            options=[""] + ETF_UNIVERSE,
            index=ETF_UNIVERSE.index(default_t) + 1 if default_t else 0,
            key=f"ticker_{i}",
        )
        weight = c2.number_input(
            "Wt %",
            min_value=0.0,
            max_value=100.0,
            value=float(default_w),
            step=1.0,
            key=f"weight_{i}",
        )
        if ticker:
            holdings.append(ticker)
            weights_input.append(weight)

    st.sidebar.subheader("Benchmark")
    benchmark = st.sidebar.selectbox("Benchmark ETF", ETF_UNIVERSE, index=ETF_UNIVERSE.index(DEFAULT_BENCHMARK))

    st.sidebar.subheader("Rebalancing")
    rebal_choice = st.sidebar.selectbox("Frequency", list(REBALANCE_OPTIONS.keys()), index=0)
    rebal_freq = REBALANCE_OPTIONS[rebal_choice]

    st.sidebar.subheader("Date Range")
    default_start = dt.date.today() - dt.timedelta(days=365 * 5)
    default_end = dt.date.today()
    start_date = st.sidebar.date_input("Start", value=default_start)
    end_date = st.sidebar.date_input("End", value=default_end)

    st.sidebar.markdown("---")
    total_weight = sum(weights_input)
    st.sidebar.metric("Total Weight", f"{total_weight:.1f}%")
    if abs(total_weight - 100) > 0.1 and total_weight > 0:
        if st.sidebar.button("🔧 Normalize Weights"):
            f = 100.0 / total_weight
            for i in range(len(weights_input)):
                st.session_state[f"weight_{i}"] = weights_input[i] * f
            st.rerun()

    if len(holdings) < 1:
        st.warning("⚠️ Please select at least 1 ETF to analyze.")
        st.stop()
    if abs(total_weight - 100) > 5.0 and total_weight > 0:
        st.warning(f"⚠️ Weights sum to {total_weight:.1f}%. Please normalize or adjust to ≈100%.")

    all_tickers = list(set(holdings + [benchmark]))
    with st.spinner("Loading price data..."):
        prices = load_prices(all_tickers, start_date, end_date)
    if prices.empty:
        st.error("❌ No price data loaded. Check tickers and date range.")
        st.stop()

    missing = set(all_tickers) - set(prices.columns)
    if missing:
        st.warning(f"⚠️ Missing data for: {', '.join(sorted(missing))}")

    holdings_available = [h for h in holdings if h in prices.columns]
    if not holdings_available:
        st.error("❌ None of the selected holdings have valid data.")
        st.stop()
    if benchmark not in prices.columns:
        st.error(f"❌ Benchmark {benchmark} has no valid data.")
        st.stop()

    weights_dict: Dict[str, float] = {}
    valid_weights = [weights_input[i] for i, h in enumerate(holdings) if h in holdings_available]
    tot_valid = sum(valid_weights)
    if tot_valid > 0:
        for t in holdings_available:
            original_idx = holdings.index(t)
            weights_dict[t] = weights_input[original_idx] / tot_valid
    else:
        st.error("❌ Total weight is zero.")
        st.stop()

    with st.spinner("Computing portfolio performance..."):
        port_equity = compute_equity_curve(prices[holdings_available], weights_dict, rebal_freq)
        bench_equity = prices[benchmark] / prices[benchmark].iloc[0]
        combined = pd.DataFrame({"Portfolio": port_equity, "Benchmark": bench_equity}).dropna()

    if combined.shape[0] < 5:
        st.error("Not enough overlapping data to compute stats. Try a wider date range.")
        st.stop()

    port_r = combined["Portfolio"].pct_change().dropna()
    bench_r = combined["Benchmark"].pct_change().dropna()
    if len(port_r) < 5 or len(bench_r) < 5:
        st.error("Insufficient data points for risk stats. Try a wider date range.")
        st.stop()

    port_stats = perf_stats(combined["Portfolio"])
    bench_stats = perf_stats(combined["Benchmark"])
    te = tracking_error(port_r, bench_r)
    ir = information_ratio(port_r, bench_r)
    beta, corr_coef = beta_corr(port_r, bench_r)

    # Charts
    st.subheader("📈 Cumulative Performance (%)")
    cum_pct = combined.subtract(1.0)
    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct["Portfolio"], name="Portfolio", line=dict(color="#2ecc71", width=2.5)))
    cum_fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct["Benchmark"], name=benchmark, line=dict(color="#3498db", width=2)))
    cum_fig.update_layout(height=H(450, 320), hovermode="x unified", yaxis_title="Cumulative Return (%)", margin=dict(l=10, r=10, t=30, b=10))
    cum_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(cum_fig, use_container_width=True)

    st.subheader("📊 Rolling 1-Year Volatility (%)")
    port_roll_vol = port_r.rolling(252).std() * np.sqrt(252) * 100
    bench_roll_vol = bench_r.rolling(252).std() * np.sqrt(252) * 100
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(x=port_roll_vol.index, y=port_roll_vol, name="Portfolio", line=dict(color="#9b59b6")))
    vol_fig.add_trace(go.Scatter(x=bench_roll_vol.index, y=bench_roll_vol, name=benchmark, line=dict(color="#34495e")))
    vol_fig.update_layout(height=H(350, 280), hovermode="x unified", yaxis_title="Annualized Volatility (%)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(vol_fig, use_container_width=True)

    st.subheader("📋 Performance Statistics")
    if IS_MOBILE:
        st.markdown("**Portfolio**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CAGR", f"{port_stats['CAGR']:.2%}")
            st.metric("Sharpe Ratio", f"{port_stats['Sharpe']:.2f}")
        with col2:
            st.metric("Ann. Vol", f"{port_stats['Ann. Vol']:.2%}")
            st.metric("Max Drawdown", f"{port_stats['Max DD']:.2%}")
        st.markdown(f"**Benchmark ({benchmark})**")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("CAGR", f"{bench_stats['CAGR']:.2%}")
            st.metric("Sharpe Ratio", f"{bench_stats['Sharpe']:.2f}")
        with col4:
            st.metric("Ann. Vol", f"{bench_stats['Ann. Vol']:.2%}")
            st.metric("Max Drawdown", f"{bench_stats['Max DD']:.2%}")
        st.markdown("**Active (vs Benchmark)**")
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Tracking Error", f"{te:.2%}")
            st.metric("Beta", f"{beta:.2f}")
        with col6:
            st.metric("Information Ratio", f"{ir:.2f}")
            st.metric("Correlation", f"{corr_coef:.2f}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Portfolio**")
            st.metric("CAGR", f"{port_stats['CAGR']:.2%}")
            st.metric("Ann. Vol", f"{port_stats['Ann. Vol']:.2%}")
            st.metric("Sharpe Ratio", f"{port_stats['Sharpe']:.2f}")
            st.metric("Max Drawdown", f"{port_stats['Max DD']:.2%}")
        with col2:
            st.markdown(f"**Benchmark ({benchmark})**")
            st.metric("CAGR", f"{bench_stats['CAGR']:.2%}")
            st.metric("Ann. Vol", f"{bench_stats['Ann. Vol']:.2%}")
            st.metric("Sharpe Ratio", f"{bench_stats['Sharpe']:.2f}")
            st.metric("Max Drawdown", f"{bench_stats['Max DD']:.2%}")
        with col3:
            st.markdown("**Active (vs Benchmark)**")
            st.metric("Tracking Error", f"{te:.2%}")
            st.metric("Information Ratio", f"{ir:.2f}")
            st.metric("Beta", f"{beta:.2f}")
            st.metric("Correlation", f"{corr_coef:.2f}")

    st.subheader("🔗 Correlation Matrix (daily returns)")
    if IS_MOBILE:
        c1 = st.container()
        c2 = st.container()
    else:
        c1, c2 = st.columns(2)
    window_choice = c1.selectbox("Window", ["Full period", "1Y (252d)", "3Y (756d)"], index=1)
    include_bench = c2.checkbox("Include benchmark in matrix", value=True)
    corr_tickers = holdings_available.copy()
    if include_bench and benchmark not in corr_tickers:
        corr_tickers.append(benchmark)

    rets_all = prices[corr_tickers].pct_change().dropna()
    if window_choice == "1Y (252d)" and len(rets_all) > 252:
        rets = rets_all.iloc[-252:].copy()
    elif window_choice == "3Y (756d)" and len(rets_all) > 756:
        rets = rets_all.iloc[-756:].copy()
    else:
        rets = rets_all

    if rets.shape[1] >= 2 and not rets.empty:
        corr = rets.corr().round(2)
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale=[[0.0, "#0b3d91"], [0.5, "#1b8c72"], [1.0, "#9fd5a2"]], zmin=-1, zmax=1, colorbar=dict(title="ρ")))
        heat.update_layout(title="Correlation Matrix", height=H(420, 320), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Need at least two series with overlapping data to compute correlation.")

    st.subheader("📉 Drawdowns")
    port_dd = drawdown_series(combined["Portfolio"])
    bench_dd = drawdown_series(combined["Benchmark"])
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=port_dd.index, y=port_dd * 100, name="Portfolio", fill="tozeroy", line=dict(color="#e74c3c")))
    dd_fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd * 100, name=benchmark, fill="tozeroy", line=dict(color="#95a5a6")))
    dd_fig.update_layout(height=H(350, 280), hovermode="x unified", yaxis_title="Drawdown (%)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(dd_fig, use_container_width=True)

    st.subheader("🥧 Portfolio Allocation")
    if IS_MOBILE:
        col_pie = st.container()
        col_table = st.container()
    else:
        col_pie, col_table = st.columns([1, 1])

    with col_pie:
        pie_df = pd.DataFrame({"Ticker": list(weights_dict.keys()), "Weight": [w * 100 for w in weights_dict.values()]})
        pie_fig = px.pie(pie_df, values="Weight", names="Ticker", title="Target Allocation", hole=0.4)
        pie_fig.update_traces(textposition="inside", textinfo="percent+label")
        pie_fig.update_layout(height=H(400, 320), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_table:
        st.markdown("**Current Weights**")
        weights_path = compute_weight_path(prices[holdings_available], weights_dict, rebal_freq)
        current_weights = weights_path.iloc[-1].to_dict() if len(weights_path) else {}
        wt = pd.DataFrame({"Ticker": list(weights_dict.keys()), "Target (%)": [w * 100 for w in weights_dict.values()], "Current (%)": [current_weights.get(t, 0) * 100 for t in weights_dict.keys()]})
        wt["Drift (%)"] = wt["Current (%)"] - wt["Target (%)"]
        st.dataframe(wt, use_container_width=True, hide_index=True)

# ===========================
# TAB 2: ETF PERFORMANCE
# ===========================
with tabs[1]:
    st.subheader("📈 ETF Performance Reference Tables")
    st.caption("Comprehensive performance metrics for all ETFs in the universe")
    
    univ_start = dt.date(2010, 1, 1)
    univ_end = end_date if 'end_date' in locals() else dt.date.today()
    
    with st.spinner("Loading ETF universe data..."):
        prices_univ = load_universe_prices(univ_start, univ_end)
    
    if prices_univ.empty:
        st.error("Could not load universe data")
    else:
        latest_date = prices_univ.index[-1]
        st.caption(f"Data as of: **{latest_date.date()}** | ETFs: **{len(prices_univ.columns)}**")
        
        # TABLE A: Trailing Returns
        st.markdown("### 📊 Trailing Returns")
        
        today = prices_univ.index[-1]
        month_start = today.replace(day=1)
        mtd = period_return(prices_univ, month_start, today)
        
        quarter = (today.month - 1) // 3
        quarter_start = today.replace(month=quarter*3 + 1, day=1)
        qtd = period_return(prices_univ, quarter_start, today)
        
        cagr_1y = trailing_cagr(prices_univ, 1, today)
        cagr_2y = trailing_cagr(prices_univ, 2, today)
        cagr_3y = trailing_cagr(prices_univ, 3, today)
        cagr_4y = trailing_cagr(prices_univ, 4, today)
        cagr_5y = trailing_cagr(prices_univ, 5, today)
        
        trailing_df = pd.DataFrame({
            "Ticker": prices_univ.columns,
            "MTD": mtd.values,
            "QTD": qtd.values,
            "1Y": cagr_1y.values,
            "2Y": cagr_2y.values,
            "3Y": cagr_3y.values,
            "4Y": cagr_4y.values,
            "5Y": cagr_5y.values,
        })
        
        trailing_df = trailing_df.dropna(subset=["MTD"]).sort_values("MTD", ascending=False)
        
        styled_trailing = trailing_df.style.format({
            "MTD": "{:.2%}", "QTD": "{:.2%}",
            "1Y": "{:.2%}", "2Y": "{:.2%}", "3Y": "{:.2%}",
            "4Y": "{:.2%}", "5Y": "{:.2%}"
        })
        
        if IS_MOBILE and len(trailing_df) > 10:
            st.dataframe(styled_trailing.data.head(10), use_container_width=True, hide_index=True)
            with st.expander("📋 Show all ETFs"):
                st.dataframe(styled_trailing, use_container_width=True, hide_index=True)
        else:
            st.dataframe(styled_trailing, use_container_width=True, hide_index=True)
        
        # TABLE B: Calendar Year Returns
        st.markdown("### 📅 Calendar Year Returns")
        
        cal_returns = calendar_year_returns(prices_univ)
        
        if not cal_returns.empty:
            # Custom color styling without matplotlib dependency
            def color_negative_red(val):
                """Color cells: red for negative, green for positive."""
                if pd.isna(val):
                    return ''
                try:
                    num_val = float(val) if isinstance(val, (int, float)) else 0
                    if num_val < -0.2:
                        return 'background-color: #d32f2f; color: white;'
                    elif num_val < -0.1:
                        return 'background-color: #e57373; color: white;'
                    elif num_val < 0:
                        return 'background-color: #ffcdd2;'
                    elif num_val > 0.2:
                        return 'background-color: #388e3c; color: white;'
                    elif num_val > 0.1:
                        return 'background-color: #81c784; color: white;'
                    elif num_val > 0:
                        return 'background-color: #c8e6c9;'
                    else:
                        return ''
                except:
                    return ''
            
            styled_cal = cal_returns.style.format("{:.1%}").applymap(color_negative_red)
            
            if IS_MOBILE:
                with st.expander("📅 View calendar returns", expanded=False):
                    st.dataframe(styled_cal, use_container_width=True)
            else:
                st.dataframe(styled_cal, use_container_width=True)
        else:
            st.info("No calendar year data available")
        
        # TABLE C: Long-term Risk Metrics
        st.markdown("### 📉 Long-term Risk Metrics")
        st.caption("Computed over full available history")
        
        risk_metrics = long_term_risk_metrics(prices_univ, bench="SPY")
        
        if not risk_metrics.empty:
            styled_risk = risk_metrics.style.format({
                "Ann. Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max DD": "{:.2%}",
                "% Down Months": "{:.1%}",
                "Worst Month": "{:.2%}",
                "Best Month": "{:.2%}",
                "Beta": "{:.2f}",
                "Corr": "{:.2f}"
            })
            
            if IS_MOBILE and len(risk_metrics) > 10:
                st.dataframe(styled_risk.data.head(10), use_container_width=True, hide_index=True)
                with st.expander("📊 Show all ETFs"):
                    st.dataframe(styled_risk, use_container_width=True, hide_index=True)
            else:
                st.dataframe(styled_risk, use_container_width=True, hide_index=True)
        else:
            st.info("No risk metrics available")
        
        # Quick stats summary
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best MTD", f"{trailing_df['MTD'].max():.2%}" if not trailing_df.empty else "N/A")
        with col2:
            st.metric("Worst MTD", f"{trailing_df['MTD'].min():.2%}" if not trailing_df.empty else "N/A")
        with col3:
            avg_vol = risk_metrics["Ann. Vol"].mean() if not risk_metrics.empty else 0
            st.metric("Avg Vol", f"{avg_vol:.2%}")
        with col4:
            avg_sharpe = risk_metrics["Sharpe"].mean() if not risk_metrics.empty else 0
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")

# Footer
st.markdown("---")
st.caption("📊 Zorroh Portfolio Analyzer · Data: Yahoo Finance · For educational purposes only")
