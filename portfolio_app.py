"""
📊 Zorroh Portfolio Analyzer — Analyzer + ETF Performance
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

# ============================================
# PERFORMANCE OPTIMIZATIONS (ADD AT TOP)
# ============================================

# 1. Reduce Streamlit overhead
st.set_page_config(
    page_title="Zorroh Portfolio Analyzer",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 2. Cache EVERYTHING aggressively
import functools

# Global cache wrapper
def ultra_cache(ttl=24*3600):
    """Cache with 24-hour default TTL"""
    return st.cache_data(show_spinner=False, ttl=ttl)

# 3. Lazy imports (only import when needed)
def lazy_import():
    """Import heavy libraries only when first needed"""
    global np, pd, yf, go, px
    if 'np' not in globals():
        import numpy as np
        import pandas as pd
        import yfinance as yf
        import plotly.graph_objects as go
        import plotly.express as px
        
# Call lazy import at first use
lazy_import()

# 4. Preload critical data in background
@ultra_cache(ttl=6*3600)
def preload_universe():
    """Preload universe data on startup"""
    try:
        import datetime as dt
        from pathlib import Path
        
        # Check for parquet cache first
        if Path("data/prices.parquet").exists():
            return True
    except:
        pass
    return False

# Run preload (non-blocking)
_ = preload_universe()

# ----------------------
# Query-param helpers (no JS)
# ----------------------
def _qp_get() -> dict:
    try:
        qp = st.query_params
        if hasattr(qp, "to_dict"):
            return qp.to_dict()
        return dict(qp)
    except Exception:
        pass
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}

def _qp_set(**kwargs):
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
    qp = _qp_get()
    v = qp.get("view")
    if isinstance(v, list):
        v = v[0] if v else None
    return v

def set_view_param(view: str):
    _qp_set(view=view)
    st.session_state["_view_mode"] = view

def get_view_mode_with_default(default: str = "desktop") -> str:
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
st.set_page_config(
    page_title="Zorroh Portfolio Analyzer",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

VIEW_MODE = get_view_mode_with_default(default="desktop")
IS_MOBILE = (VIEW_MODE == "mobile")

# Global CSS
st.markdown(
    """
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:100%}
[data-testid="stMetricValue"]{font-size:1.1rem}
[data-testid="stMetricDelta"]{font-size:0.9rem}
table{width:100%}
.view-toggle-container{display:flex;justify-content:center;align-items:center;margin:1rem 0;gap:.5rem}
@media (max-width:1024px){
  .block-container{padding-left:.6rem;padding-right:.6rem;padding-top:.5rem}
  [data-testid="stMetricValue"]{font-size:1rem}
  [data-testid="stMetricDelta"]{font-size:.85rem}
  .stPlotlyChart{height:auto!important}
  table{font-size:.85rem}
  button{min-height:44px}
  [data-testid="stSelectbox"]{font-size:.9rem}
}
@media (min-width:768px) and (max-width:1024px){
  .block-container{padding-left:1rem;padding-right:1rem}
}
@media (min-width:1025px){
  .block-container{padding-left:2rem;padding-right:2rem}
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------
# Data / universe
# ----------------------
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
# Price loading with optional parquet
# ----------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_prices(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    if isinstance(start, dt.date) and isinstance(end, dt.date) and start > end:
        start, end = end, start

    want = sorted(set(tickers))
    cached = _try_read_cached_prices()
    pieces, covered = [], set()

    if cached is not None:
        c = cached.copy()
        c = c.loc[(c.index >= pd.Timestamp(start)) & (c.index <= pd.Timestamp(end))]
        c = _sanitize_columns(c)
        have = [t for t in want if t in c.columns]
        if have:
            pieces.append(c[have]); covered.update(have)

    missing = [t for t in want if t not in covered]
    if missing:
        try:
            df = yf.download(missing, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            fetched = _yf_to_wide(df, missing)
            if not fetched.empty: pieces.append(fetched)
        except Exception as e:
            st.warning(f"Live fetch failed for {missing}: {e}")

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1)
    out = _sanitize_columns(out)
    out = out.reindex(columns=want).dropna(how="all")
    return out

def _try_read_cached_prices() -> pd.DataFrame | None:
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
    df = df.copy(); df.columns = new_cols
    return df

def _yf_to_wide(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
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
# Portfolio mechanics & stats
# ----------------------
def get_rebalance_dates(prices: pd.DataFrame, freq: str) -> List[pd.Timestamp]:
    if freq is None: return []
    idx = prices.index
    if len(idx) == 0: return []
    if freq == "M":
        candidates = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="MS")
    elif freq == "Q":
        candidates = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="QS")
    elif freq == "Y":
        candidates = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="YS")
    else:
        return []
    aligned = [idx[0]] if len(idx) else []
    for d in candidates:
        pos = idx.searchsorted(d)
        if pos < len(idx): aligned.append(idx[pos])
    return sorted(set(aligned))

def compute_equity_curve(prices: pd.DataFrame, weights: Dict[str, float], rebal_freq: str) -> pd.Series:
    prices = prices.dropna(how="all")
    rets = prices.pct_change().fillna(0.0)
    rebal_dates = set(get_rebalance_dates(prices, rebal_freq))
    pv = pd.Series(index=prices.index, dtype=float)
    if len(prices.index) == 0: return pv
    pv.iloc[0] = 1.0
    current = weights.copy()
    for i in range(1, len(prices)):
        d = prices.index[i]
        if rebal_freq is not None and d in rebal_dates:
            current = weights.copy()
        port_ret = sum(current.get(t, 0) * rets.iloc[i].get(t, 0.0) for t in weights)
        pv.iloc[i] = pv.iloc[i-1] * (1 + port_ret)
        if rebal_freq is None or d not in rebal_dates:
            tot = sum(current.get(t, 0) * (1 + rets.iloc[i].get(t, 0.0)) for t in weights)
            if tot > 0:
                current = {t: current.get(t, 0) * (1 + rets.iloc[i].get(t, 0.0)) / tot for t in weights}
    return pv

def drawdown_series(series: pd.Series) -> pd.Series:
    return (series - series.cummax()) / series.cummax()

def max_drawdown(series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    cum_max = series.cummax()
    dd = (series - cum_max) / cum_max
    if dd.empty or dd.isna().all():
        return 0.0, (series.index[0] if len(series) else pd.NaT), (series.index[0] if len(series) else pd.NaT)
    trough = dd.idxmin(); peak = series.loc[:trough].idxmax()
    return float(dd.min()), peak, trough

def perf_stats(equity: pd.Series) -> Dict[str, float]:
    r = equity.pct_change().dropna()
    if len(r) < 30:
        return {"CAGR": 0.0, "Ann. Vol": 0.0, "Sharpe": 0.0, "Max DD": 0.0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0.0
    vol = r.std() * np.sqrt(252)
    sharpe = (cagr / vol) if vol > 0 else 0.0  # rf ~ 0 for simplicity
    mdd, _, _ = max_drawdown(equity)
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe, "Max DD": mdd}

def tracking_error(port_r: pd.Series, bench_r: pd.Series) -> float:
    a = (port_r - bench_r).dropna()
    return a.std() * np.sqrt(252)

def information_ratio(port_r: pd.Series, bench_r: pd.Series) -> float:
    a = (port_r - bench_r).dropna()
    te = a.std() * np.sqrt(252)
    return ((a.mean() * 252) / te) if te > 0 else 0.0

def beta_corr_monthly(asset: pd.Series, bench: pd.Series) -> Tuple[float, float]:
    a_m = asset.resample("M").last().pct_change().dropna()
    b_m = bench.resample("M").last().pct_change().dropna()
    aligned = pd.DataFrame({"a": a_m, "b": b_m}).dropna()
    if len(aligned) < 24 or aligned["b"].var() <= 0:
        return np.nan, np.nan
    cov = aligned.cov()
    beta = cov.loc["a", "b"] / cov.loc["b", "b"]
    corr = aligned["a"].corr(aligned["b"])
    return float(beta), float(corr)

def up_down_capture(asset: pd.Series, bench: pd.Series) -> Tuple[float, float]:
    a_m = asset.resample("M").last().pct_change().dropna()
    b_m = bench.resample("M").last().pct_change().dropna()
    aligned = pd.DataFrame({"a": a_m, "b": b_m}).dropna()
    if aligned.empty:
        return np.nan, np.nan
    up = aligned[aligned["b"] > 0]; down = aligned[aligned["b"] < 0]
    up_cap = up["a"].mean() / up["b"].mean() if len(up) else np.nan
    down_cap = down["a"].mean() / down["b"].mean() if len(down) else np.nan
    return float(up_cap), float(down_cap)

# ----------------------
# ETF performance helpers
# ----------------------
def period_return(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    ret = pd.Series(index=df.columns, dtype=float)
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            ret[col] = np.nan; continue
        s_idx = series.index.searchsorted(start_dt)
        e_idx = series.index.searchsorted(end_dt)
        s_idx = min(max(s_idx, 0), len(series)-1)
        e_idx = min(max(e_idx, 0), len(series)-1)
        if s_idx >= e_idx:
            ret[col] = np.nan
        else:
            ret[col] = (series.iloc[e_idx] / series.iloc[s_idx]) - 1
    return ret

def trailing_cagr(df: pd.DataFrame, years: int, end_dt: pd.Timestamp) -> pd.Series:
    start_dt = end_dt - relativedelta(years=years)
    ret = pd.Series(index=df.columns, dtype=float)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0: ret[col] = np.nan; continue
        s_idx = s.index.searchsorted(start_dt)
        e_idx = s.index.searchsorted(end_dt)
        s_idx = min(max(s_idx, 0), len(s)-1)
        e_idx = min(max(e_idx, 0), len(s)-1)
        if s_idx >= e_idx:
            ret[col] = np.nan; continue
        sv = s.iloc[s_idx]; ev = s.iloc[e_idx]
        yrs = (s.index[e_idx] - s.index[s_idx]).days / 365.25
        if yrs < years * 0.8:  # require ~80% of requested period
            ret[col] = np.nan
        else:
            ret[col] = (ev / sv) ** (1/yrs) - 1
    return ret

def calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0: continue
        row = {"Ticker": col}
        years = sorted(s.index.year.unique())
        for y in years:
            ydata = s[s.index.year == y]
            if len(ydata) >= 2:
                row[str(y)] = (ydata.iloc[-1] / ydata.iloc[0]) - 1
        curr = s.index[-1].year
        ytd = s[s.index.year == curr]
        if len(ytd) >= 2:
            row["YTD"] = (ytd.iloc[-1] / ytd.iloc[0]) - 1
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("Ticker")
    return out

def long_term_risk_metrics(df: pd.DataFrame, bench: str = "SPY") -> pd.DataFrame:
    results = []
    bench_series = df[bench].dropna() if bench in df.columns else pd.Series(dtype=float)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 30: continue

        d = s.pct_change().dropna()
        m_series = s.resample("M").last()
        m = m_series.pct_change().dropna()

        row = {"Ticker": col}
        row["Ann. Vol"] = (d.std() * np.sqrt(252)) if len(d) else np.nan
        row["Sharpe"] = ((d.mean()*252) / (d.std()*np.sqrt(252))) if len(d) and d.std() > 0 else np.nan

        eq = s / s.iloc[0]
        dd = (eq / eq.cummax()) - 1
        row["Max DD"] = dd.min() if len(dd) else np.nan

        row["% Down Months"] = (m < 0).sum() / len(m) if len(m) else np.nan
        row["Worst Month"] = m.min() if len(m) else np.nan
        row["Best Month"] = m.max() if len(m) else np.nan

        # Sortino (daily downside deviation)
        downside = d[d < 0]
        if len(downside) > 0:
            downside_std = downside.std() * np.sqrt(252)
            row["Sortino"] = (d.mean() * 252) / downside_std if downside_std > 0 else np.nan
        else:
            row["Sortino"] = np.nan

        # Benchmark-dependent metrics (monthly)
        if bench in df.columns and col != bench and len(bench_series) > 0:
            beta, corr = beta_corr_monthly(s, bench_series)
            up, down = up_down_capture(s, bench_series)
            row["Beta (vs SPY)"] = beta
            row["Corr (vs SPY)"] = corr
            row["Up Capture (vs SPY)"] = up
            row["Down Capture (vs SPY)"] = down
        else:
            row["Beta (vs SPY)"] = 1.0 if col == bench else np.nan
            row["Corr (vs SPY)"] = 1.0 if col == bench else np.nan
            row["Up Capture (vs SPY)"] = 1.0 if col == bench else np.nan
            row["Down Capture (vs SPY)"] = 1.0 if col == bench else np.nan

        results.append(row)
    return pd.DataFrame(results)

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_universe_prices(start: dt.date, end: dt.date) -> pd.DataFrame:
    return load_prices(ETF_UNIVERSE, start, end)

# ----------------------
# UI — Tabs
# ----------------------
st.title("📊 Zorroh Portfolio Analyzer")
st.caption("Build and analyze diversified ETF portfolios vs a benchmark")

tabs = st.tabs(["📊 Analyzer", "📈 ETF Performance"])

# Common view toggle renderer
def render_view_toggle():
    st.markdown('<div class="view-toggle-container">', unsafe_allow_html=True)
    choice = st.radio(
        "View Mode",
        options=["📱 Mobile", "🖥️ Desktop"],
        index=0 if IS_MOBILE else 1,
        horizontal=True,
        key=f"view_toggle_{st.session_state.get('_tab', '0')}",
        label_visibility="collapsed",
    )
    new_view = "mobile" if "Mobile" in choice else "desktop"
    if new_view != VIEW_MODE:
        set_view_param(new_view)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def H(desktop: int, mobile: int) -> int:
    return mobile if IS_MOBILE else desktop

# ===========================
# TAB 1: ANALYZER
# ===========================
with tabs[0]:
    st.session_state["_tab"] = "analyzer"
    render_view_toggle()

    if Path("data/prices.parquet").exists():
        st.caption("⚡ Using precomputed data from `data/prices.parquet`")
    else:
        st.caption("⏳ No precomputed parquet found; will fetch live data")

    # Sidebar controls
    st.sidebar.header("Portfolio Setup")
    st.sidebar.subheader("Holdings (max 5)")
    holdings, weights_input = [], []
    for i in range(5):
        c1, c2 = st.sidebar.columns([3, 2])
        default_t = DEFAULT_HOLDINGS[i] if i < len(DEFAULT_HOLDINGS) else None
        default_w = DEFAULT_WEIGHTS[i] if i < len(DEFAULT_WEIGHTS) else 0
        ticker = c1.selectbox(
            f"ETF {i+1}",
            options=[""] + ETF_UNIVERSE,
            index=ETF_UNIVERSE.index(default_t) + 1 if default_t else 0,
            key=f"ticker_{i}",
        )
        weight = c2.number_input(
            "Wt %",
            min_value=0.0, max_value=100.0,
            value=float(default_w), step=1.0, key=f"weight_{i}",
        )
        if ticker:
            holdings.append(ticker); weights_input.append(weight)

    st.sidebar.subheader("Benchmark")
    benchmark = st.sidebar.selectbox("Benchmark ETF", ETF_UNIVERSE, index=ETF_UNIVERSE.index(DEFAULT_BENCHMARK))

    st.sidebar.subheader("Rebalancing")
    rebal_choice = st.sidebar.selectbox("Frequency", list(REBALANCE_OPTIONS.keys()), index=0)
    rebal_freq = REBALANCE_OPTIONS[rebal_choice]

    st.sidebar.subheader("Date Range")
    default_start = dt.date.today() - dt.timedelta(days=365*5)
    default_end = dt.date.today()
    start_date = st.sidebar.date_input("Start", value=default_start)
    end_date   = st.sidebar.date_input("End", value=default_end)

    st.sidebar.markdown("---")
    total_weight = sum(weights_input)
    st.sidebar.metric("Total Weight", f"{total_weight:.1f}%")
    if abs(total_weight - 100) > 0.1 and total_weight > 0:
        if st.sidebar.button("🔧 Normalize Weights"):
            f = 100.0 / total_weight
            for i in range(len(weights_input)):
                st.session_state[f"weight_{i}"] = weights_input[i] * f
            st.rerun()

    # Data validation
    if len(holdings) < 1:
        st.warning("⚠️ Please select at least 1 ETF to analyze."); st.stop()
    if abs(total_weight - 100) > 5.0 and total_weight > 0:
        st.warning(f"⚠️ Weights sum to {total_weight:.1f}%. Please normalize or adjust to ≈100%.")

    all_tickers = list(set(holdings + [benchmark]))
    with st.spinner("Loading price data..."):
        prices = load_prices(all_tickers, start_date, end_date)
    if prices.empty:
        st.error("❌ No price data loaded. Check tickers and date range."); st.stop()

    missing = set(all_tickers) - set(prices.columns)
    if missing:
        st.warning(f"⚠️ Missing data for: {', '.join(sorted(missing))}")

    holdings_available = [h for h in holdings if h in prices.columns]
    if not holdings_available:
        st.error("❌ None of the selected holdings have valid data."); st.stop()
    if benchmark not in prices.columns:
        st.error(f"❌ Benchmark {benchmark} has no valid data."); st.stop()

    weights_dict: Dict[str, float] = {}
    valid_weights = [weights_input[i] for i, h in enumerate(holdings) if h in holdings_available]
    tot_valid = sum(valid_weights)
    if tot_valid > 0:
        for t in holdings_available:
            original_idx = holdings.index(t)
            weights_dict[t] = weights_input[original_idx] / tot_valid
    else:
        st.error("❌ Total weight is zero."); st.stop()

    with st.spinner("Computing portfolio performance..."):
        port_equity = compute_equity_curve(prices[holdings_available], weights_dict, rebal_freq)
        bench_equity = prices[benchmark] / prices[benchmark].iloc[0]
        combined = pd.DataFrame({"Portfolio": port_equity, "Benchmark": bench_equity}).dropna()

    if combined.shape[0] < 5:
        st.error("Not enough overlapping data to compute stats. Try a wider date range."); st.stop()

    port_r  = combined["Portfolio"].pct_change().dropna()
    bench_r = combined["Benchmark"].pct_change().dropna()
    if len(port_r) < 5 or len(bench_r) < 5:
        st.error("Insufficient data points for risk stats. Try a wider date range."); st.stop()

    port_stats  = perf_stats(combined["Portfolio"])
    bench_stats = perf_stats(combined["Benchmark"])
    te = tracking_error(port_r, bench_r)
    ir = information_ratio(port_r, bench_r)

    # Charts
    st.subheader("📈 Cumulative Performance (%)")
    cum_pct = combined.subtract(1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct["Portfolio"], name="Portfolio",
                             line=dict(color="#2ecc71", width=2.5),
                             hovertemplate="%{y:.2%}<extra>Portfolio</extra>"))
    fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct["Benchmark"], name=benchmark,
                             line=dict(color="#3498db", width=2),
                             hovertemplate=f"%{{y:.2%}}<extra>{benchmark}</extra>"))
    fig.update_layout(height=H(450,320), hovermode="x unified", yaxis_title="Cumulative Return (%)",
                      margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Rolling 1-Year Volatility (%)")
    port_roll = port_r.rolling(252).std() * np.sqrt(252) * 100
    bench_roll = bench_r.rolling(252).std() * np.sqrt(252) * 100
    vfig = go.Figure()
    vfig.add_trace(go.Scatter(x=port_roll.index,  y=port_roll,  name="Portfolio", line=dict(color="#9b59b6")))
    vfig.add_trace(go.Scatter(x=bench_roll.index, y=bench_roll, name=benchmark,  line=dict(color="#34495e")))
    vfig.update_layout(height=H(350,280), hovermode="x unified", yaxis_title="Annualized Volatility (%)",
                       margin=dict(l=10,r=10,t=30,b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(vfig, use_container_width=True)

    st.subheader("📋 Performance Statistics")
    if IS_MOBILE:
        st.markdown("**Portfolio**")
        c1,c2 = st.columns(2)
        with c1:
            st.metric("CAGR", f"{port_stats['CAGR']:.2%}")
            st.metric("Sharpe Ratio", f"{port_stats['Sharpe']:.2f}")
        with c2:
            st.metric("Ann. Vol", f"{port_stats['Ann. Vol']:.2%}")
            st.metric("Max Drawdown", f"{port_stats['Max DD']:.2%}")
        st.markdown(f"**Benchmark ({benchmark})**")
        c3,c4 = st.columns(2)
        with c3:
            st.metric("CAGR", f"{bench_stats['CAGR']:.2%}")
            st.metric("Sharpe Ratio", f"{bench_stats['Sharpe']:.2f}")
        with c4:
            st.metric("Ann. Vol", f"{bench_stats['Ann. Vol']:.2%}")
            st.metric("Max Drawdown", f"{bench_stats['Max DD']:.2%}")
        st.markdown("**Active (vs Benchmark)**")
        c5,c6 = st.columns(2)
        with c5:
            st.metric("Tracking Error", f"{te:.2%}")
        with c6:
            st.metric("Information Ratio", f"{ir:.2f}")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Portfolio**")
            st.metric("CAGR", f"{port_stats['CAGR']:.2%}")
            st.metric("Ann. Vol", f"{port_stats['Ann. Vol']:.2%}")
            st.metric("Sharpe Ratio", f"{port_stats['Sharpe']:.2f}")
            st.metric("Max Drawdown", f"{port_stats['Max DD']:.2%}")
        with c2:
            st.markdown(f"**Benchmark ({benchmark})**")
            st.metric("CAGR", f"{bench_stats['CAGR']:.2%}")
            st.metric("Ann. Vol", f"{bench_stats['Ann. Vol']:.2%}")
            st.metric("Sharpe Ratio", f"{bench_stats['Sharpe']:.2f}")
            st.metric("Max Drawdown", f"{bench_stats['Max DD']:.2%}")
        with c3:
            st.markdown("**Active (vs Benchmark)**")
            st.metric("Tracking Error", f"{te:.2%}")
            st.metric("Information Ratio", f"{ir:.2f}")

    # ----------------------
    # Correlation Matrix (daily returns) — with annotations toggle
    # ----------------------
    st.subheader("🔗 Correlation Matrix (daily returns)")

    if IS_MOBILE:
        cc1, cc2 = st.container(), st.container()
    else:
        cc1, cc2 = st.columns(2)

    window_choice = cc1.selectbox("Window", ["Full period", "1Y (252d)", "3Y (756d)"], index=1)
    include_bench = cc2.checkbox("Include benchmark in matrix", value=True)

    corr_tickers = [*holdings_available]
    if include_bench and benchmark not in corr_tickers:
        corr_tickers.append(benchmark)

    rets_all = prices[corr_tickers].pct_change().dropna()
    if window_choice == "1Y (252d)" and len(rets_all) > 252:
        rets = rets_all.iloc[-252:]
    elif window_choice == "3Y (756d)" and len(rets_all) > 756:
        rets = rets_all.iloc[-756:]
    else:
        rets = rets_all

    if rets.shape[1] >= 2 and not rets.empty:
        corr = rets.corr().round(2)

        show_vals = cc2.checkbox("Show values in cells", value=(corr.shape[0] <= 10))

        heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1, zmax=1,
                colorscale=[[0.0, "#0b3d91"], [0.5, "#1b8c72"], [1.0, "#9fd5a2"]],
                colorbar=dict(title="ρ"),
                hovertemplate="%{x} ↔ %{y}<br>ρ = %{z:.2f}<extra></extra>",
            )
        )

        anns = []
        if show_vals:
            for i, y in enumerate(corr.index):
                for j, x in enumerate(corr.columns):
                    val = corr.iloc[i, j]
                    if pd.isna(val):
                        continue
                    txt_color = "white" if abs(val) > 0.35 else "black"
                    anns.append(dict(x=x, y=y, text=f"{val:.2f}", showarrow=False,
                                     font=dict(color=txt_color, size=12)))

        heat.update_layout(
            title="Correlation Matrix",
            height=H(420, 320),
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            annotations=anns,
        )
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Need at least two series with overlapping data to compute correlation.")

    st.subheader("📉 Drawdowns")
    port_dd = drawdown_series(combined["Portfolio"])
    bench_dd = drawdown_series(combined["Benchmark"])
    dfig = go.Figure()
    dfig.add_trace(go.Scatter(x=port_dd.index, y=port_dd*100, name="Portfolio", fill="tozeroy", line=dict(color="#e74c3c")))
    dfig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd*100, name=benchmark, fill="tozeroy", line=dict(color="#95a5a6")))
    dfig.update_layout(height=H(350,280), hovermode="x unified", yaxis_title="Drawdown (%)",
                       margin=dict(l=10,r=10,t=30,b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(dfig, use_container_width=True)

# ===========================
# TAB 2: ETF PERFORMANCE
# ===========================
with tabs[1]:
    st.session_state["_tab"] = "perf"
    render_view_toggle()

    st.subheader("📈 ETF Performance Reference Tables")
    st.caption("Comprehensive performance metrics for the ETF universe")

    # Universe period
    univ_start = dt.date(2010, 1, 1)
    univ_end = dt.date.today()

    with st.spinner("Loading ETF universe data..."):
        prices_univ = load_universe_prices(univ_start, univ_end)

    if prices_univ.empty:
        st.error("Could not load universe data")
    else:
        first_date = prices_univ.index[0]
        latest_date = prices_univ.index[-1]
        st.caption(f"Data window: **{first_date.date()} → {latest_date.date()}** · ETFs: **{len(prices_univ.columns)}**")

        # ---- Trailing Returns
        st.markdown("### 📊 Trailing Returns")
        st.caption("Compare short-term (MTD/QTD) and long-term (1–5Y CAGR) trends across ETFs.")

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
        }).dropna(subset=["MTD"])
        trailing_df = trailing_df.sort_values("MTD", ascending=False)

        st.dataframe(
            trailing_df.style.format({
                "MTD":"{:.2%}","QTD":"{:.2%}",
                "1Y":"{:.2%}","2Y":"{:.2%}","3Y":"{:.2%}",
                "4Y":"{:.2%}","5Y":"{:.2%}"
            }),
            use_container_width=True, hide_index=True
        )

        # ---- Calendar Year Returns
        st.markdown("### 📅 Calendar Year Returns")
        st.caption("Annual heatmap highlights rotation between asset classes — darker = stronger signal.")
        cal = calendar_year_returns(prices_univ)
        if not cal.empty:
            # Order columns: YTD, then years descending (latest→2010)
            year_cols = [c for c in cal.columns if c != "YTD" and str(c).isdigit()]
            ordered = (["YTD"] if "YTD" in cal.columns else []) + sorted(year_cols, key=int, reverse=True)
            cal = cal.reindex(columns=ordered)

            # Darker gradient heat styling
            def cy_heat(val):
                if pd.isna(val): return ""
                try: v = float(val)
                except: return ""
                reds  = ["#4a0000","#6a0000","#8b0000","#b71c1c","#ffcdd2"]
                greens= ["#e8f5e9","#a5d6a7","#66bb6a","#2e7d32","#0f3d1f"]
                if v <= -0.40: return f"background-color:{reds[0]};color:#fff"
                if v <= -0.30: return f"background-color:{reds[1]};color:#fff"
                if v <= -0.20: return f"background-color:{reds[2]};color:#fff"
                if v <= -0.10: return f"background-color:{reds[3]};color:#fff"
                if v <   0.00: return f"background-color:{reds[4]};color:#000"
                if v <   0.10: return f"background-color:{greens[0]};color:#000"
                if v <   0.20: return f"background-color:{greens[1]};color:#000"
                if v <   0.30: return f"background-color:{greens[2]};color:#fff"
                if v <   0.40: return f"background-color:{greens[3]};color:#fff"
                return f"background-color:{greens[4]};color:#fff"

            styled_cal = cal.style.format("{:.1%}").applymap(cy_heat)
            st.dataframe(styled_cal, use_container_width=True)
        else:
            st.info("No calendar year data available")

        # ---- Long-term Risk Metrics
        st.markdown("### 📉 Long-term Risk Metrics")
        st.caption(
            "Computed on full available history "
            f"(**{first_date.date()} → {latest_date.date()}**). "
            "Vol/Sharpe/Sortino/MaxDD use total returns; "
            "**Beta, Corr, Up/Down Capture are vs SPY using monthly returns**."
        )

        risk = long_term_risk_metrics(prices_univ, bench="SPY")
        if not risk.empty:
            risk = risk.reset_index(drop=False)
            cols_order = [
                "Ticker", "Ann. Vol", "Sharpe", "Sortino", "Max DD",
                "% Down Months", "Worst Month", "Best Month",
                "Beta (vs SPY)", "Corr (vs SPY)",
                "Up Capture (vs SPY)", "Down Capture (vs SPY)"
            ]
            show_cols = [c for c in cols_order if c in risk.columns]
            risk = risk[show_cols]

            st.dataframe(
                risk.style.format({
                    "Ann. Vol":"{:.2%}",
                    "Sharpe":"{:.2f}",
                    "Sortino":"{:.2f}",
                    "Max DD":"{:.2%}",
                    "% Down Months":"{:.1%}",
                    "Worst Month":"{:.2%}",
                    "Best Month":"{:.2%}",
                    "Beta (vs SPY)":"{:.2f}",
                    "Corr (vs SPY)":"{:.2f}",
                    "Up Capture (vs SPY)":"{:.2f}",
                    "Down Capture (vs SPY)":"{:.2f}",
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No risk metrics available")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best MTD", f"{trailing_df['MTD'].max():.2%}" if not trailing_df.empty else "N/A")
        with c2:
            st.metric("Worst MTD", f"{trailing_df['MTD'].min():.2%}" if not trailing_df.empty else "N/A")
        with c3:
            avg_vol = risk["Ann. Vol"].mean() if not risk.empty else 0
            st.metric("Avg Ann. Vol", f"{avg_vol:.2%}")
        with c4:
            avg_sh = risk["Sharpe"].mean() if not risk.empty else 0
            st.metric("Avg Sharpe", f"{avg_sh:.2f}")
# ----------------------
# ETF Reference
# ----------------------
st.markdown("### 📚 ETF Reference (Universe)")
ETF_CATALOG = {
    "SPY": ("SPDR S&P 500 ETF Trust", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "QQQ": ("Invesco QQQ Trust", "https://www.invesco.com/us/financial-products/etfs/product-detail?productId=QQQ"),
    "IWM": ("iShares Russell 2000 ETF", "https://www.ishares.com/us/products/239710/iwm-ishares-russell-2000-etf"),
    "EFA": ("iShares MSCI EAFE ETF", "https://www.ishares.com/us/products/239612/efa-ishares-msci-eafe-etf"),
    "EFA": ("iShares MSCI EAFE ETF", "https://www.blackrock.com/ae/intermediaries/products/239623/ishares-msci-eafe-etf"),
    "EEM": ("iShares MSCI Emerging Markets ETF", "https://www.ishares.com/us/products/239637/eem-ishares-msci-emerging-markets-etf"),
    "AGG": ("iShares Core U.S. Aggregate Bond ETF", "https://www.ishares.com/us/products/239458/ishares-core-us-aggregate-bond-etf"),
    "BND": ("Vanguard Total Bond Market ETF", "https://investor.vanguard.com/investment-products/etfs/profile/bnd"),
    "LQD": ("iShares iBoxx $ Investment Grade Corporate Bond ETF", "https://www.ishares.com/us/products/239566/lqd-ishares-iboxx-investment-grade-corporate-bond-etf"),
    "HYG": ("iShares iBoxx $ High Yield Corporate Bond ETF", "https://www.ishares.com/us/products/239565/hyg-ishares-iboxx-high-yield-corporate-bond-etf"),
    "TLT": ("iShares 20+ Year Treasury Bond ETF", "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf"),
    "IEF": ("iShares 7-10 Year Treasury Bond ETF", "https://www.ishares.com/us/products/239455/ishares-710-year-treasury-bond-etf"),
    "GLD": ("SPDR Gold Shares", "https://www.spdrgoldshares.com/usa/"),
    "SLV": ("iShares Silver Trust", "https://www.ishares.com/us/products/239855/ishares-silver-trust-fund"),
    "DBC": ("Invesco DB Commodity Index Tracking Fund", "https://www.invesco.com/us/financial-products/etfs/product-detail?productId=DBC"),
    "VNQ": ("Vanguard Real Estate ETF", "https://investor.vanguard.com/investment-products/etfs/profile/vnq"),
    "XLB": ("Materials Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-materials-select-sector-spdr-fund-xlb"),
    "XLE": ("Energy Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-energy-select-sector-spdr-fund-xle"),
    "XLF": ("Financial Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-financial-select-sector-spdr-fund-xlf"),
    "XLI": ("Industrial Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-industrial-select-sector-spdr-fund-xli"),
    "XLK": ("Technology Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/technology-select-sector-spdr-fund-xlk"),
    "XLP": ("Consumer Staples Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-consumer-staples-select-sector-spdr-fund-xlp"),
    "XLRE": ("Real Estate Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-real-estate-select-sector-spdr-fund-xlre"),
    "XLU": ("Utilities Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-utilities-select-sector-spdr-fund-xlu"),
    "XLV": ("Health Care Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-health-care-select-sector-spdr-fund-xlv"),
    "XLY": ("Consumer Discretionary Select Sector SPDR Fund", "https://www.ssga.com/us/en/etfs/funds/the-consumer-discretionary-select-sector-spdr-fund-xly"),
}
def render_etf_catalog(etf_map: dict):
    rows = []
    for t, (name, url) in etf_map.items():
        label = f"[{name}]({url})" if url else name
        rows.append({"Ticker": t, "Name / Page": label})
    df = pd.DataFrame(rows).sort_values("Ticker")
    md = "| Ticker | Name / Page |\n|---|---|\n"
    for _, r in df.iterrows():
        md += f"| {r['Ticker']} | {r['Name / Page']} |\n"
    st.markdown(md, unsafe_allow_html=True)
render_etf_catalog(ETF_CATALOG)


# ----------------------
# Footer + Disclosures
# ----------------------
st.markdown("---")
st.caption("📊 Zorroh Portfolio Analyzer · Data: Yahoo Finance · For educational purposes only")

with st.expander("ℹ️ Methodology & Disclosures", expanded=False):
    st.markdown(
        """
**Data source**  
All ETF data via Yahoo Finance daily adjusted prices. If `data/prices.parquet` exists, it is used to speed things up.

**Return calculations**  
- **MTD/QTD**: Simple total return from start of month/quarter to latest date.  
- **1–5Y**: CAGR using adjusted close; requires sufficient history.  
- **Calendar-year returns**: Simple total return for each calendar year; YTD uses current year to latest date.  

**Risk metrics**  
- **Annualized Volatility**: Std dev of daily returns × √252.  
- **Sharpe**: Annualized return ÷ annualized volatility (rf ≈ 0).  
- **Sortino**: Annualized return ÷ downside deviation (daily).  
- **Max Drawdown**: Minimum value of (equity / rolling peak − 1) over history.  
- **Beta, Correlation, Up/Down Capture**: Computed **vs SPY** using **monthly returns** for stability.  

**Important**  
Historical data may differ from total-return indices due to dividend timing/splits.  
Past performance is **not** indicative of future results. This tool is for **educational purposes only** and is **not** investment advice.
"""
    )
