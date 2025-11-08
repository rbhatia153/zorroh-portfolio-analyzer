"""
📊 Zorroh Portfolio Analyzer — Mobile-optimized
ETF portfolio construction and analysis tool for teaching diversification.

REQUIREMENTS:
- streamlit>=1.37.0
- yfinance>=0.2.50
- pandas>=2.2.0
- numpy>=1.26.0
- plotly>=5.22.0

Run: streamlit run portfolio_app.py
"""

import datetime as dt
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------
# Configuration
# ----------------------
st.set_page_config(
    page_title="Zorroh Portfolio Analyzer",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed"  # nicer on mobile
)

# Global CSS: slightly smaller fonts on narrow screens, responsive tables
st.markdown("""
<style>
/* tighten paddings */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.1rem; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
/* tables scale down a bit on phones */
table { width: 100%; }
@media (max-width: 768px) {
  .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
  [data-testid="stMetricValue"] { font-size: 1rem; }
  [data-testid="stMetricDelta"] { font-size: 0.85rem; }
  .stPlotlyChart { height: auto !important; }
  table { font-size: 0.85rem; }
}
</style>
""", unsafe_allow_html=True)

ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "BND", "LQD", "HYG", "TLT",
    "IEF", "GLD", "SLV", "DBC", "VNQ", "XLB", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY"
]

REBALANCE_OPTIONS = {
    "Buy & Hold (No Rebalance)": None,
    "Monthly": "M",
    "Quarterly": "Q",
    "Annual": "Y"
}

DEFAULT_HOLDINGS = ["SPY", "EFA", "AGG", "VNQ", "GLD"]
DEFAULT_WEIGHTS = [40, 20, 25, 10, 5]
DEFAULT_BENCHMARK = "SPY"

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Load adjusted close prices from Yahoo Finance."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers, start=start, end=end,
            interval="1d", auto_adjust=True, progress=False
        )
        if data.empty:
            return pd.DataFrame()
        if len(tickers) == 1:
            if isinstance(data, pd.Series):
                prices = data.to_frame(name=tickers[0])
            else:
                prices = data[['Close']].copy()
                prices.columns = [tickers[0]]
        else:
            prices = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data
        return prices.dropna(how='all')
    except Exception as e:
        st.error(f"Error loading prices: {e}")
        return pd.DataFrame()

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
    for d in candidates:
        pos = idx.searchsorted(d)
        if pos < len(idx):
            aligned.append(idx[pos])
    return aligned

def compute_equity_curve(prices: pd.DataFrame, weights: Dict[str, float], rebal_freq: str) -> pd.Series:
    """Portfolio equity curve with optional rebalancing."""
    returns = prices.pct_change().fillna(0)
    rebal_dates = set(get_rebalance_dates(prices, rebal_freq))
    pv = pd.Series(index=prices.index, dtype=float)
    pv.iloc[0] = 1.0
    current = weights.copy()
    for i in range(1, len(prices)):
        d = prices.index[i]
        if rebal_freq is not None and d in rebal_dates:
            current = weights.copy()
        port_ret = sum(current.get(t, 0) * returns.iloc[i][t] for t in weights if t in returns.columns)
        pv.iloc[i] = pv.iloc[i-1] * (1 + port_ret)
        if rebal_freq is None or d not in rebal_dates:
            tot = sum(current.get(t, 0) * (1 + returns.iloc[i][t]) for t in weights if t in returns.columns)
            if tot > 0:
                current = {t: current.get(t, 0) * (1 + returns.iloc[i][t]) / tot for t in weights if t in returns.columns}
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
        return 0.0, series.index[0], series.index[0]
    trough = dd.idxmin()
    peak = series.loc[:trough].idxmax()
    return float(dd.min()), peak, trough

def drawdown_series(series: pd.Series) -> pd.Series:
    return (series - series.cummax()) / series.cummax()

def perf_stats(equity: pd.Series) -> Dict[str, float]:
    r = equity.pct_change().dropna()
    if len(r) == 0:
        return {"CAGR": 0.0, "Ann. Vol": 0.0, "Sharpe": 0.0, "Max DD": 0.0}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0.0
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
# UI
# ----------------------
st.title("📊 Zorroh Portfolio Analyzer")
st.caption("Build and analyze diversified ETF portfolios vs a benchmark")

# 👉 Display controls (mobile optimization)
st.sidebar.header("Display")
compact = st.sidebar.toggle("📱 Compact (mobile) mode", value=False,
                            help="Stacks layouts, shrinks chart heights & fonts for phones.")
def H(desktop:int, mobile:int) -> int:
    return mobile if compact else desktop

# ----------------------
# Sidebar: Portfolio Setup
# ----------------------
st.sidebar.header("Portfolio Setup")
st.sidebar.subheader("Holdings (max 5)")
holdings, weights_input = [], []
for i in range(5):
    c1, c2 = st.sidebar.columns([3, 2])
    default_t = DEFAULT_HOLDINGS[i] if i < len(DEFAULT_HOLDINGS) else None
    default_w = DEFAULT_WEIGHTS[i] if i < len(DEFAULT_WEIGHTS) else 0
    ticker = c1.selectbox(f"ETF {i+1}", options=[""] + ETF_UNIVERSE,
                          index=ETF_UNIVERSE.index(default_t) + 1 if default_t else 0, key=f"ticker_{i}")
    weight = c2.number_input("Wt %", min_value=0.0, max_value=100.0, value=float(default_w),
                             step=1.0, key=f"weight_{i}")
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

# ----------------------
# Data
# ----------------------
if len(holdings) < 1:
    st.warning("⚠️ Please select at least 1 ETF to analyze."); st.stop()
if abs(total_weight - 100) > 5.0:
    st.warning(f"⚠️ Weights sum to {total_weight:.1f}%. Please normalize or adjust to ≈100%.")

all_tickers = list(set(holdings + [benchmark]))
with st.spinner("Loading price data..."):
    prices = load_prices(all_tickers, start_date, end_date)
if prices.empty:
    st.error("❌ No price data loaded. Check tickers and date range."); st.stop()

missing = set(all_tickers) - set(prices.columns)
if missing:
    st.warning(f"⚠️ Missing data for: {', '.join(missing)}")

holdings_available = [h for h in holdings if h in prices.columns]
if not holdings_available:
    st.error("❌ None of the selected holdings have valid data."); st.stop()
if benchmark not in prices.columns:
    st.error(f"❌ Benchmark {benchmark} has no valid data."); st.stop()

weights_dict = {}
valid_weights = [weights_input[i] for i, h in enumerate(holdings) if h in holdings_available]
tot_valid = sum(valid_weights)
if tot_valid > 0:
    for t in holdings_available:
        original_idx = holdings.index(t)
        weights_dict[t] = weights_input[original_idx] / tot_valid
else:
    st.error("❌ Total weight is zero."); st.stop()

# ----------------------
# Compute
# ----------------------
with st.spinner("Computing portfolio performance..."):
    port_equity = compute_equity_curve(prices[holdings_available], weights_dict, rebal_freq)
    bench_equity = prices[benchmark] / prices[benchmark].iloc[0]
    combined = pd.DataFrame({'Portfolio': port_equity, 'Benchmark': bench_equity}).dropna()
    if combined.empty:
        st.error("❌ No overlapping data between portfolio and benchmark."); st.stop()
    port_r  = combined['Portfolio'].pct_change().dropna()
    bench_r = combined['Benchmark'].pct_change().dropna()

port_stats = perf_stats(combined['Portfolio'])
bench_stats = perf_stats(combined['Benchmark'])
te = tracking_error(port_r, bench_r)
ir = information_ratio(port_r, bench_r)
beta, corr_coef = beta_corr(port_r, bench_r)

# ----------------------
# Charts & Tables
# ----------------------

# 1) Cumulative Performance (%)
st.subheader("📈 Cumulative Performance (%)")
cum_pct = combined.subtract(1.0)
cum_fig = go.Figure()
cum_fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct['Portfolio'], name='Portfolio',
                             line=dict(color='#2ecc71', width=2.5),
                             hovertemplate='%{y:.2%}<extra>Portfolio</extra>'))
cum_fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct['Benchmark'], name=benchmark,
                             line=dict(color='#3498db', width=2),
                             hovertemplate='%{y:.2%}<extra>'+benchmark+'</extra>'))
cum_fig.update_layout(
    height=H(450, 320), hovermode='x unified',
    yaxis_title='Cumulative Return (%)', xaxis_title='Date',
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
cum_fig.update_yaxes(tickformat=".0%")
st.plotly_chart(cum_fig, use_container_width=True)

# 2) Rolling 1-Year Volatility
st.subheader("📊 Rolling 1-Year Volatility (%)")
port_roll_vol  = port_r.rolling(252).std() * np.sqrt(252) * 100
bench_roll_vol = bench_r.rolling(252).std() * np.sqrt(252) * 100
vol_fig = go.Figure()
vol_fig.add_trace(go.Scatter(x=port_roll_vol.index,  y=port_roll_vol,  name='Portfolio', line=dict(color='#9b59b6')))
vol_fig.add_trace(go.Scatter(x=bench_roll_vol.index, y=bench_roll_vol, name=benchmark,  line=dict(color='#34495e')))
vol_fig.update_layout(
    height=H(350, 280), hovermode='x unified',
    yaxis_title='Annualized Volatility (%)', xaxis_title='Date',
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(vol_fig, use_container_width=True)

# 3) Key Statistics (stack on mobile)
st.subheader("📋 Performance Statistics")
if compact:
    cols = [st.container()]
else:
    cols = st.columns(3)

with cols[0]:
    st.markdown("**Portfolio**")
    st.metric("CAGR", f"{port_stats['CAGR']:.2%}")
    st.metric("Ann. Vol", f"{port_stats['Ann. Vol']:.2%}")
    st.metric("Sharpe Ratio", f"{port_stats['Sharpe']:.2f}")
    st.metric("Max Drawdown", f"{port_stats['Max DD']:.2%}")

if not compact:
    c2, c3 = cols[1], cols[2]
else:
    c2 = st.container(); c3 = st.container()

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
    st.metric("Beta", f"{beta:.2f}")
    st.metric("Correlation", f"{corr_coef:.2f}")

# 4) Correlation Matrix
st.subheader("🔗 Correlation Matrix (daily returns)")
c1, c2 = st.columns(2) if not compact else (st.container(), st.container())
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
    heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[
            [0.0, "#0b3d91"], [0.25, "#146b8a"],
            [0.5, "#1b8c72"], [0.75, "#3fa37a"], [1.0, "#9fd5a2"]
        ],
        zmin=-1, zmax=1, zmid=0, colorbar=dict(title="ρ"),
        hovertemplate='%{x} ↔ %{y}<br>ρ = %{z:.2f}<extra></extra>'
    ))
    ann = []
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            ann.append(dict(x=c, y=r, text=f"{corr.iloc[i, j]:.2f}", showarrow=False, font=dict(color="black")))
    heat.update_layout(
        title="Correlation Matrix (Zorroh Palette)",
        height=H(420, 320), margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="white"),
        annotations=ann
    )
    st.plotly_chart(heat, use_container_width=True)
    st.caption(f"As of {rets.index[-1].date()} — window: **{window_choice}**")
    st.dataframe(corr, use_container_width=True)
else:
    st.info("Need at least two series with overlapping data to compute correlation.")

# 5) Drawdowns
st.subheader("📉 Drawdowns")
port_dd = drawdown_series(combined['Portfolio'])
bench_dd = drawdown_series(combined['Benchmark'])
dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(x=port_dd.index, y=port_dd * 100, name='Portfolio', fill='tozeroy', line=dict(color='#e74c3c')))
dd_fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd * 100, name=benchmark,   fill='tozeroy', line=dict(color='#95a5a6')))
dd_fig.update_layout(
    height=H(350, 280), hovermode='x unified',
    yaxis_title='Drawdown (%)', xaxis_title='Date',
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(dd_fig, use_container_width=True)

# 6) Allocation (stack on mobile)
st.subheader("🥧 Portfolio Allocation")
if compact:
    col_pie = st.container(); col_table = st.container()
else:
    col_pie, col_table = st.columns([1, 1])

with col_pie:
    pie_df = pd.DataFrame({'Ticker': list(weights_dict.keys()),
                           'Weight': [w * 100 for w in weights_dict.values()]})
    pie_fig = px.pie(pie_df, values='Weight', names='Ticker', title='Target Allocation', hole=0.4)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(height=H(400, 320), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)

with col_table:
    st.markdown("**Current Weights**")
    weights_path = compute_weight_path(prices[holdings_available], weights_dict, rebal_freq)
    current_weights = weights_path.iloc[-1].to_dict()
    wt = pd.DataFrame({
        'Ticker': list(weights_dict.keys()),
        'Target (%)': [w * 100 for w in weights_dict.values()],
        'Current (%)': [current_weights.get(t, 0) * 100 for t in weights_dict.keys()],
    })
    wt['Drift (%)'] = wt['Current (%)'] - wt['Target (%)']

    def _drift_style(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v): styles.append("")
            elif v < -0.25: styles.append("background-color:#1e8449; color:white;")
            elif v < -0.10: styles.append("background-color:#27ae60; color:white;")
            elif v >  0.25: styles.append("background-color:#c0392b; color:white;")
            elif v >  0.10: styles.append("background-color:#e74c3c; color:white;")
            else: styles.append("")
        return styles

    styled = (wt.style
              .format({'Target (%)': '{:.1f}', 'Current (%)': '{:.1f}', 'Drift (%)': '{:.1f}'})
              .apply(_drift_style, subset=['Drift (%)']))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("**🎨 Drift Color Legend**")
    st.markdown("""
    <table style="width:100%; border-collapse: collapse; text-align:left;">
      <tr><th style="padding:6px; border-bottom:1px solid #ddd;">Color</th>
          <th style="padding:6px; border-bottom:1px solid #ddd;">Meaning</th></tr>
      <tr><td style="background-color:#1e8449; color:white; padding:6px;">Dark Green</td>
          <td style="padding:6px;">Strongly underweight (≤ −25% vs target)</td></tr>
      <tr><td style="background-color:#27ae60; color:white; padding:6px;">Medium Green</td>
          <td style="padding:6px;">Slightly underweight (−10% to −25%)</td></tr>
      <tr><td style="background-color:#e74c3c; color:white; padding:6px;">Medium Red</td>
          <td style="padding:6px;">Slightly overweight (+10% to +25%)</td></tr>
      <tr><td style="background-color:#c0392b; color:white; padding:6px;">Dark Red</td>
          <td style="padding:6px;">Strongly overweight (≥ +25% vs target)</td></tr>
      <tr><td style="background-color:#f4f6f6; color:black; padding:6px;">None</td>
          <td style="padding:6px;">Near target (within ±10%)</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ----------------------
# ETF Reference
# ----------------------
st.markdown("### 📚 ETF Reference (Universe)")
ETF_CATALOG = {
    "SPY": ("SPDR S&P 500 ETF Trust", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "QQQ": ("Invesco QQQ Trust", "https://www.invesco.com/us/financial-products/etfs/product-detail?productId=QQQ"),
    "IWM": ("iShares Russell 2000 ETF", "https://www.ishares.com/us/products/239710/iwm-ishares-russell-2000-etf"),
    "EFA": ("iShares MSCI EAFE ETF", "https://www.ishares.com/us/products/239612/efa-ishares-msci-eafe-etf"),
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
# Footer
# ----------------------
st.markdown("---")
st.caption("📊 Zorroh Portfolio Analyzer · Data: Yahoo Finance · For educational purposes only")