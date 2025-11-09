z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0.0, "#0b3d91"],
                [0.25, "#146b8a"],
                [0.5, "#1b8c72"],
                [0.75, "#3fa37a"],
                [1.0, "#9fd5a2"],
            ],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar=dict(title="ρ"),
            hovertemplate="%{x} ↔ %{y}<br>ρ = %{z:.2f}<extra></extra>",
        )
    )
    ann = []
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            ann.append(
                dict(x=c, y=r, text=f"{corr.iloc[i, j]:.2f}", showarrow=False, font=dict(color="black"))
            )
    heat.update_layout(
        title="Correlation Matrix (Zorroh Palette)",
        height=H(420, 320),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        annotations=ann,
    )
    st.plotly_chart(heat, use_container_width=True)
    st.caption(f"As of {rets.index[-1].date()} — window: **{window_choice}**")

    # Mobile: show correlation table in expander to save space
    if IS_MOBILE and len(corr) > 5:
        with st.expander("📊 View Correlation Table", expanded=False):
            st.dataframe(corr, use_container_width=True)
    else:
        st.dataframe(corr, use_container_width=True)
else:
    st.info("Need at least two series with overlapping data to compute correlation.")

# 5) Drawdowns
st.subheader("📉 Drawdowns")
port_dd = drawdown_series(combined["Portfolio"])
bench_dd = drawdown_series(combined["Benchmark"])
dd_fig = go.Figure()
dd_fig.add_trace(
    go.Scatter(x=port_dd.index, y=port_dd * 100, name="Portfolio", fill="tozeroy", line=dict(color="#e74c3c"))
)
dd_fig.add_trace(
    go.Scatter(x=bench_dd.index, y=bench_dd * 100, name=benchmark, fill="tozeroy", line=dict(color="#95a5a6"))
)
dd_fig.update_layout(
    height=H(350, 280),
    hovermode="x unified",
    yaxis_title="Drawdown (%)",
    xaxis_title="Date",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(dd_fig, use_container_width=True)

# 6) Allocation (responsive layout)
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
    wt = pd.DataFrame(
        {
            "Ticker": list(weights_dict.keys()),
            "Target (%)": [w * 100 for w in weights_dict.values()],
            "Current (%)": [current_weights.get(t, 0) * 100 for t in weights_dict.keys()],
        }
    )
    wt["Drift (%)"] = wt["Current (%)"] - wt["Target (%)"]

    def _drift_style(col: pd.Series):
        styles = []
        for v in col:
            if pd.isna(v):
                styles.append("")
            elif v < -0.25:
                styles.append("background-color:#1e8449; color:white;")
            elif v < -0.10:
                styles.append("background-color:#27ae60; color:white;")
            elif v > 0.25:
                styles.append("background-color:#c0392b; color:white;")
            elif v > 0.10:
                styles.append("background-color:#e74c3c; color:white;")
            else:
                styles.append("")
        return styles

    styled = (
        wt.style.format({"Target (%)": "{:.1f}", "Current (%)": "{:.1f}", "Drift (%)": "{:.1f}"}).apply(
            _drift_style, subset=["Drift (%)"]
        )
    )
    try:
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(
            wt.style.format({"Target (%)": "{:.1f}", "Current (%)": "{:.1f}", "Drift (%)": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**🎨 Drift Color Legend**")
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

# ----------------------
# ETF Reference
# ----------------------
st.markdown("### 📚 ETF Reference (Universe)")
ETF_CATALOG = {
    "SPY": ("SPDR S&P 500 ETF Trust", "https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy"),
    "QQQ": ("Invesco QQQ Trust", "https://www.invesco.com/us/financial-products/etfs/product-detail?productId=QQQ"),
    "IWM": ("iShares Russell 2000 ETF", "https://www.ishares.com/us/products/239710/iwm-ishares-russell-2000-etf"),
    "EFA": ("iShares MSCI EAFE ETF", "https://www.blackrock.com/ae/intermediaries/products/239623/ishares-msci-eafe-etf"),
    "EEM": ("iShares MSCI Emerging Markets ETF", "https://www.ishares.com/us/products/239637/eem-ishares-msci-emerging-markets-etf"),
    "AGG": ("iShares Core U.S. Aggregate Bond ETF", "https://www.ishares.com/us/products/239458/ishares-core-us-aggregate-bond-etf"),
    "BND": ("Vanguard Total Bond Market ETF", "https://www.morningstar.com/etfs/xnas/bnd/portfolio"),
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

def render_etf_catalog(etf_map: dict, is_mobile: bool = False):
    """Render ETF catalog with mobile optimization."""
    rows = []
    for t, (name, url) in etf_map.items():
        label = f"[{name}]({url})" if url else name
        rows.append({"Ticker": t, "Name / Page": label})
    df = pd.DataFrame(rows).sort_values("Ticker")
    md = "| Ticker | Name / Page |\n|---|---|\n"
    for _, r in df.iterrows():
        md += f"| {r['Ticker']} | {r['Name / Page']} |\n"

    # On mobile, show in collapsible expander to save space
    if is_mobile:
        with st.expander("📖 View Full ETF Catalog", expanded=False):
            st.markdown(md, unsafe_allow_html=True)
    else:
        st.markdown(md, unsafe_allow_html=True)

render_etf_catalog(ETF_CATALOG, is_mobile=IS_MOBILE)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("📊 Zorroh Portfolio Analyzer · Data: Yahoo Finance · For educational purposes only")
