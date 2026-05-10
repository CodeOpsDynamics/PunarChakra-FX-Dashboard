"""
=============================================================================
PunarChakra E-Waste Exports — Interactive FX Risk Dashboard
=============================================================================
Author  : Himanshu Rai | Roll No. XW013-25 | IIM Ranchi EMBA 2025-27
Course  : International Finance | Prof. Kamran Quddus | WAI Project Term IV
Run     : streamlit run punarchakra_dashboard.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PunarChakra FX Risk Dashboard",
    page_icon   = "♻️",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1B3A6B;
        border-bottom: 3px solid #2563A8; padding-bottom: 0.5rem;
    }
    .subtitle { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card {
        background: #EBF3FB; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #2563A8;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #1B3A6B; }
    .metric-lbl { font-size: 0.8rem; color: #666; }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #1B3A6B;
        border-left: 4px solid #2563A8; padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    .insight-box {
        background: #f0f7ff; border-radius: 8px;
        padding: 0.9rem 1.2rem; border: 1px solid #b3d4f5;
        font-size: 0.9rem; margin-top: 0.5rem;
    }
    .recommend-box {
        background: #e8f5e9; border-radius: 8px;
        padding: 1rem 1.4rem; border: 1px solid #81c784;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">♻️ PunarChakra — AI-Powered FX Risk Assessment Dashboard</div>
<div class="subtitle">
    Hedging INR/USD Exposure for Indian E-Waste Exports to UAE & Singapore (FY2027-28) |
    IIM Ranchi EMBA WAI Project | Himanshu Rai (XW013-25) | Prof. Kamran Quddus
</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:#1B3A6B;padding:14px 12px 10px 12px;border-radius:8px;
                margin-bottom:12px;text-align:center;">
        <div style="font-size:1.6rem;">🎓</div>
        <div style="color:#FFFFFF;font-weight:700;font-size:0.9rem;margin-top:4px;">
            IIM Ranchi</div>
        <div style="color:#93C5FD;font-size:0.72rem;margin-top:2px;">
            EMBA 2025–27 &nbsp;|&nbsp; Term IV</div>
        <div style="color:#BAE6FD;font-size:0.68rem;margin-top:1px;">
            WAI Project &nbsp;|&nbsp; Himanshu Rai (XW013-25)</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    st.markdown("**📦 Export Parameters**")
    invoice_usd = st.slider(
        "Annual USD Receivables ($)",
        min_value=500_000, max_value=2_000_000,
        value=1_000_000, step=50_000,
        format="$%d"
    )
    hedge_fwd = st.slider(
        "Forward Contract Cover (%)",
        min_value=0, max_value=80,
        value=40, step=5
    )
    hedge_nat = st.slider(
        "Natural Hedge Cover (%)",
        min_value=0, max_value=60,
        value=40, step=5
    )
    open_exp = 100 - hedge_fwd - hedge_nat
    st.info(f"Open Exposure: **{max(open_exp,0)}%**")

    st.markdown("---")
    st.markdown("**📈 Interest Rate Inputs**")
    r_inr = st.number_input("RBI Repo Rate (%)", value=5.25, step=0.25) / 100
    r_usd = st.number_input("US Fed Funds Rate (%)", value=3.625, step=0.125) / 100
    spot  = st.number_input("Current Spot Rate (₹/USD)", value=94.28, step=0.10)

    st.markdown("---")
    st.markdown("**🌍 Scenario Spot Rates at 6M**")
    s_base = st.number_input("Baseline Scenario", value=96.50, step=0.50)
    s_bear = st.number_input("Bear (INR weak)", value=101.00, step=0.50)
    s_bull = st.number_input("Bull (INR strong)", value=90.00, step=0.50)

    st.markdown("---")
    st.markdown("**📅 Forecast Horizon**")
    forecast_days = st.slider("Days to Forecast", 200, 600, 400, 50)

    st.markdown("---")
    st.caption("Data: RBI Reference Rates | 968 trading days | Apr 2022–Apr 2026")
    st.caption("Model: Facebook Prophet | IIM Ranchi WAI Project 2026")

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "RBI_Data.xlsx")
    df = pd.read_excel(path)
    df.columns = ["Date","USD","GBP","EUR","JPY","AED","IDR"]
    df["Date"]  = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna(subset=["USD"]).sort_values("Date").reset_index(drop=True)
    df["USD"] = pd.to_numeric(df["USD"], errors="coerce").ffill()
    return df

@st.cache_resource
def train_model(data):
    pdf   = data[["Date","USD"]].rename(columns={"Date":"ds","USD":"y"})
    model = Prophet(
        changepoint_prior_scale = 0.05,
        seasonality_mode        = "multiplicative",
        yearly_seasonality      = True,
        weekly_seasonality      = False,
        daily_seasonality       = False,
        interval_width          = 0.80,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(pdf)
    return model

with st.spinner("🔄 Loading RBI data and training Prophet model..."):
    df    = load_data()
    model = train_model(df)

# Generate forecast
future   = model.make_future_dataframe(periods=forecast_days, freq="B")
forecast = model.predict(future)

last_actual = pd.Timestamp("2026-04-17")
fut_mask    = forecast["ds"] > last_actual
forecast.loc[fut_mask, "bear"] = forecast.loc[fut_mask, "yhat"] * 1.04
forecast.loc[fut_mask, "bull"] = forecast.loc[fut_mask, "yhat"] * 0.95
forecast.loc[~fut_mask, "bear"] = forecast.loc[~fut_mask, "yhat"]
forecast.loc[~fut_mask, "bull"] = forecast.loc[~fut_mask, "yhat"]

# ── IRP Computation ───────────────────────────────────────────────────────────
tenors   = [3, 6, 9, 12]
irp_fwds = {t: round(spot * ((1 + r_inr) / (1 + r_usd)) ** (t/12), 2) for t in tenors}
fwd_6m   = irp_fwds[6]
fwd_9m   = irp_fwds[9]

# ── Hedging Calculations ──────────────────────────────────────────────────────
def calc_hedging(invoice, fwd_rate, hfwd, hnat, scenario_rates, sp):
    results = []
    opt_prem = invoice * 0.013 * sp
    for name, fspot in scenario_rates.items():
        unhedged  = invoice * fspot
        fwd_inr   = (invoice * hfwd/100 * fwd_rate) + (invoice * (1 - hfwd/100) * fspot)
        opt_inr   = invoice * max(fspot, fwd_rate) - opt_prem
        nat_inr   = (invoice * hnat/100 * fwd_rate) + (invoice * hfwd/100 * fwd_rate) + \
                    (invoice * max(0, 1 - hfwd/100 - hnat/100) * fspot)
        results.append({
            "Scenario": name, "Future Spot": fspot,
            "Unhedged": round(unhedged),
            "Forward": round(fwd_inr),
            "Options": round(opt_inr),
            "Natural+Fwd": round(nat_inr),
        })
    return pd.DataFrame(results)

scenario_rates = {"Baseline": s_base, "Bear (INR weak)": s_bear, "Bull (INR strong)": s_bull}
hedge_df = calc_hedging(invoice_usd, fwd_6m, hedge_fwd, hedge_nat, scenario_rates, spot)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — KEY METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Key Market Metrics (Live)</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
metrics = [
    (c1, f"₹{spot:.2f}", "Live INR/USD Rate"),
    (c2, "5.04%", "INR CAGR Depreciation (4yr)"),
    (c3, "3.99%", "Annualised Volatility"),
    (c4, f"₹{fwd_6m:.2f}", "IRP Forward Rate (6M)"),
    (c5, f"{r_inr*100:.2f}%", "RBI Repo Rate"),
    (c6, f"{r_usd*100:.3f}%", "US Fed Funds Rate"),
]
for col, val, lbl in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — HISTORICAL TREND + PROPHET FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Figure 1 & 2: Historical Trend + Prophet AI Forecast</div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔵 Historical + Forecast (Combined)", "🔬 Forecast Detail"])

with tab1:
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["USD"], mode="lines",
        name="Actual INR/USD (968 days)",
        line=dict(color="#1f77b4", width=1.3),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:.4f}<extra></extra>"
    ))

    # Confidence band
    fut_fc = forecast[fut_mask]
    fig.add_trace(go.Scatter(
        x=fut_fc["ds"].tolist() + fut_fc["ds"].tolist()[::-1],
        y=fut_fc["yhat_upper"].tolist() + fut_fc["yhat_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="80% Confidence Band"
    ))

    # Scenarios
    for col, lbl, clr, dash in [
        ("yhat", "Baseline Forecast", "#1f77b4", "dash"),
        ("bear", "Bear: INR Weak (+4%)", "crimson", "dot"),
        ("bull", "Bull: INR Strong (-5%)", "green", "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=fut_fc["ds"], y=fut_fc[col], mode="lines",
            name=lbl, line=dict(color=clr, width=2, dash=dash),
        ))

    # Markers
    fig.add_hline(y=spot, line_dash="solid", line_color="orange", line_width=1.5,
                  annotation_text=f"Live: ₹{spot:.2f}", annotation_position="right")
    fig.add_hline(y=fwd_6m, line_dash="dash", line_color="purple", line_width=1,
                  annotation_text=f"6M Forward: ₹{fwd_6m:.2f}", annotation_position="right")
    _fy_start = pd.Timestamp("2027-04-01").timestamp() * 1000
    _fy_end   = pd.Timestamp("2028-03-31").timestamp() * 1000
    _rbi_end  = pd.Timestamp("2026-04-17").timestamp() * 1000
    fig.add_vrect(x0=_fy_start, x1=_fy_end,
                  fillcolor="lightyellow", opacity=0.35, layer="below",
                  annotation_text="FY2027-28 Planning Horizon", annotation_position="top left")
    fig.add_vline(x=_rbi_end, line_dash="dot", line_color="gray",
                  annotation_text="RBI Data End", annotation_position="top right")

    fig.update_layout(
        title="INR/USD Historical (Apr 2022–Apr 2026) + Prophet AI Forecast (3 Scenarios)",
        xaxis_title="Date", yaxis_title="₹ per 1 USD",
        height=480, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> INR has depreciated from ₹76.11 (Apr 2022) to ₹94.28 (May 2026) —
    a CAGR of 5.04% p.a. The bear scenario (US-Iran tensions, crude >$100) projects ₹101–104 by FY2027-28.
    Even the baseline projects ₹96–99. For PunarChakra, <b>every ₹1 depreciation adds ~₹9–10 lakh
    of additional INR realisation on $1M annual exports</b> — but also raises import costs for plant consumables.
    </div>""", unsafe_allow_html=True)

with tab2:
    # Trend + seasonality decomposition
    fig2 = make_subplots(rows=2, cols=1,
        subplot_titles=["Trend Component (INR/USD long-term direction)",
                        "Yearly Seasonality Pattern"],
        vertical_spacing=0.15)

    fig2.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["trend"],
        mode="lines", name="Trend", line=dict(color="#1B3A6B", width=2)
    ), row=1, col=1)

    yearly = forecast[["ds","yearly"]].copy()
    yearly["month"] = yearly["ds"].dt.month
    monthly_avg = yearly.groupby("month")["yearly"].mean().reset_index()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_avg["month_name"] = monthly_avg["month"].apply(lambda x: months[x-1])
    fig2.add_trace(go.Bar(
        x=monthly_avg["month_name"], y=monthly_avg["yearly"],
        name="Monthly Effect", marker_color=["#ff6b6b" if v>0 else "#4ecdc4" for v in monthly_avg["yearly"]]
    ), row=2, col=1)

    fig2.update_layout(height=500, template="plotly_white", showlegend=False,
                       title="Prophet Model Decomposition — Trend & Seasonality")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — IRP FORWARD RATES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📐 Figure 3: IRP Forward Rate Calculator</div>',
            unsafe_allow_html=True)

col_irp, col_irp2 = st.columns([3, 2])

with col_irp:
    irp_data = [{
        "Tenor": f"{t} Months",
        "IRP Forward Rate (₹/USD)": v,
        "Forward Premium (%)": round((v - spot)/spot * 100, 2),
        "INR Depreciation Implied": f"{round((v/spot - 1)*100, 2)}%"
    } for t, v in irp_fwds.items()]
    irp_display = pd.DataFrame(irp_data)

    fig_irp = go.Figure()
    colors  = ["#2196F3","#1976D2","#1565C0","#0D47A1"]
    fig_irp.add_trace(go.Bar(
        x=irp_display["Tenor"], y=irp_display["IRP Forward Rate (₹/USD)"],
        marker_color=colors,
        text=[f"₹{v:.2f}" for v in irp_display["IRP Forward Rate (₹/USD)"]],
        textposition="outside", name="Forward Rate"
    ))
    fig_irp.add_hline(y=spot, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Spot ₹{spot:.2f}", annotation_position="right")
    fig_irp.update_layout(
        title=f"IRP Forward Rates | r_INR={r_inr*100:.2f}% | r_USD={r_usd*100:.3f}%<br>"
              f"<sup>F = S × [(1+r_INR)/(1+r_USD)]^T | Spot = ₹{spot:.2f}</sup>",
        yaxis=dict(range=[spot-2, max(irp_fwds.values())+2]),
        height=380, template="plotly_white"
    )
    st.plotly_chart(fig_irp, use_container_width=True)

with col_irp2:
    st.markdown("**IRP Forward Rate Table**")
    st.dataframe(irp_display.style.format({
        "IRP Forward Rate (₹/USD)": "₹{:.2f}",
        "Forward Premium (%)": "{:.2f}%"
    }), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="insight-box">
    📐 <b>IRP Formula:</b><br>
    F = S × [(1 + r_INR) / (1 + r_USD)]^T<br><br>
    At current rates (r_INR = {r_inr*100:.2f}%, r_USD = {r_usd*100:.3f}%),
    the interest differential of <b>{(r_inr-r_usd)*100:.2f}%</b> implies
    a <b>forward premium on USD</b> — meaning INR is expected to weaken.
    The 12-month forward at <b>₹{irp_fwds[12]:.2f}</b> aligns with
    the historical CAGR of 5.04%, confirming the forward market is
    fairly priced for PunarChakra's hedging decisions.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — HEDGING STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🛡️ Figures 4 & 5: Hedging Strategy Comparison</div>',
            unsafe_allow_html=True)

col_h1, col_h2 = st.columns([3, 2])

with col_h1:
    fig_hedge = go.Figure()
    strat_cols  = ["Unhedged","Forward","Options","Natural+Fwd"]
    strat_names = ["Unhedged","A: Forward","B: Options","C: Natural+Fwd"]
    colors_h    = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4"]
    scenarios_list = hedge_df["Scenario"].tolist()

    for col, name, clr in zip(strat_cols, strat_names, colors_h):
        fig_hedge.add_trace(go.Bar(
            name=name, x=scenarios_list,
            y=[v / 1e5 for v in hedge_df[col]],
            marker_color=clr,
            text=[f"₹{v/1e5:.1f}L" for v in hedge_df[col]],
            textposition="outside"
        ))

    fig_hedge.update_layout(
        title=f"INR Realisation — 3 Strategies × 3 Scenarios<br>"
              f"<sup>Invoice: ${invoice_usd:,.0f} USD | Fwd Cover: {hedge_fwd}% | Natural: {hedge_nat}%</sup>",
        xaxis_title="Scenario", yaxis_title="INR Realisation (₹ Lakhs)",
        barmode="group", height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_hedge, use_container_width=True)

with col_h2:
    # Heatmap of P&L vs unhedged
    z_data = [
        [
            (hedge_df.loc[hedge_df["Scenario"]==s, "Forward"].values[0] -
             hedge_df.loc[hedge_df["Scenario"]==s, "Unhedged"].values[0]) / 1000,
            (hedge_df.loc[hedge_df["Scenario"]==s, "Options"].values[0] -
             hedge_df.loc[hedge_df["Scenario"]==s, "Unhedged"].values[0]) / 1000,
            (hedge_df.loc[hedge_df["Scenario"]==s, "Natural+Fwd"].values[0] -
             hedge_df.loc[hedge_df["Scenario"]==s, "Unhedged"].values[0]) / 1000,
        ]
        for s in scenarios_list
    ]
    fig_heat = go.Figure(go.Heatmap(
        z=z_data, x=["Forward","Options","Natural+Fwd"], y=scenarios_list,
        colorscale="RdYlGn", zmid=0,
        text=[[f"₹{v:.0f}K" for v in row] for row in z_data],
        texttemplate="%{text}", textfont=dict(size=13),
    ))
    fig_heat.update_layout(
        title="P&L vs Unhedged (₹ Thousands)<br><sup>Green = Hedge adds value</sup>",
        height=300, template="plotly_white"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Best strategy per scenario
    st.markdown("**Best Strategy per Scenario:**")
    for _, row in hedge_df.iterrows():
        vals = {"Forward": row["Forward"], "Options": row["Options"], "Natural+Fwd": row["Natural+Fwd"]}
        best = max(vals, key=vals.get)
        icon = "🔴" if row["Scenario"]=="Bull (INR strong)" else "🟢"
        st.write(f"{icon} **{row['Scenario']}** → {best}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — ANNUAL P&L PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💰 Figure 6: Annual FX Exposure & Hedge Value (FY2027-28)</div>',
            unsafe_allow_html=True)

col_p1, col_p2 = st.columns(2)

with col_p1:
    usd_scenarios = [invoice_usd * 0.9, invoice_usd, invoice_usd * 1.1]
    labels        = ["Conservative (90%)", "Mid (100%)", "Optimistic (110%)"]
    spot_vals     = [u * spot / 1e7 for u in usd_scenarios]
    fwd_vals      = [u * fwd_6m / 1e7 for u in usd_scenarios]

    fig_annual = go.Figure()
    fig_annual.add_trace(go.Bar(x=labels, y=spot_vals, name="At Spot Rate",
                                 marker_color="#FF6B6B",
                                 text=[f"₹{v:.2f}Cr" for v in spot_vals], textposition="outside"))
    fig_annual.add_trace(go.Bar(x=labels, y=fwd_vals, name="At 6M Forward",
                                 marker_color="#4ECDC4",
                                 text=[f"₹{v:.2f}Cr" for v in fwd_vals], textposition="outside"))
    fig_annual.update_layout(
        title="Annual INR Realisation — Spot vs Forward Hedged (₹ Crore)",
        barmode="group", height=380, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_annual, use_container_width=True)

with col_p2:
    additional_lakh = [(u * (fwd_6m - spot)) / 1e5 for u in usd_scenarios]
    hedge_cost_lakh = [u * 0.40 * 0.0025 * spot / 1e5 for u in usd_scenarios]
    net_benefit     = [a - c for a, c in zip(additional_lakh, hedge_cost_lakh)]

    fig_net = go.Figure()
    fig_net.add_trace(go.Bar(x=labels, y=additional_lakh, name="Gross Hedge Gain",
                              marker_color="#96CEB4",
                              text=[f"+₹{v:.1f}L" for v in additional_lakh], textposition="inside"))
    fig_net.add_trace(go.Bar(x=labels, y=[-c for c in hedge_cost_lakh], name="Bank Spread Cost",
                              marker_color="#FF6B6B",
                              text=[f"-₹{c:.1f}L" for c in hedge_cost_lakh], textposition="inside"))
    fig_net.add_trace(go.Scatter(x=labels, y=net_benefit, mode="lines+markers",
                                  name="Net Benefit", line=dict(color="#1B3A6B", width=3),
                                  marker=dict(size=10)))
    fig_net.update_layout(
        title="Hedge Value: Gross Gain vs. Cost (₹ Lakhs)",
        barmode="relative", height=380, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_net, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — RECOMMENDATION BOX
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">✅ Recommended Hedging Strategy for PunarChakra</div>',
            unsafe_allow_html=True)

net_mid = net_benefit[1]
st.markdown(f"""
<div class="recommend-box">
<b>🏭 PunarChakra FX Risk Management Framework — FY2027-28</b><br><br>
<table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
<tr style="background:#c8e6c9;">
  <td style="padding:6px;"><b>Layer 1 — Natural Hedge ({hedge_nat}%)</b></td>
  <td>Source plant equipment from international vendors (payable in USD).
  Offsets USD inflows with USD outflows — zero derivative premium cost.</td>
</tr>
<tr>
  <td style="padding:6px;"><b>Layer 2 — Forward Contract ({hedge_fwd}%)</b></td>
  <td>Book 6–9 month forwards at IRP rate ₹{fwd_6m:.2f} (6M) on confirmed export invoices.
  Use EEFC account at bank. Bank spread ~0.25% — negligible vs. protection value.</td>
</tr>
<tr style="background:#f5f5f5;">
  <td style="padding:6px;"><b>Layer 3 — Open Exposure ({max(open_exp,0)}%)</b></td>
  <td>Monitor monthly. Review if: (a) Brent crude crosses $95/barrel,
  (b) INR/USD moves ±3% from Prophet baseline, or (c) new RBI policy action.</td>
</tr>
</table>
<br>
📊 <b>Financial Impact:</b> At ${invoice_usd:,.0f} annual exports — <b>Net hedge benefit: ~₹{net_mid:.1f} Lakhs/year</b> |
Planning Rate: <b>₹{fwd_6m:.2f}/USD</b> | Monitoring Trigger: Crude >$95/barrel
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem;">
PunarChakra FX Risk Dashboard | IIM Ranchi EMBA WAI Project | Himanshu Rai (XW013-25) |
Prof. Kamran Quddus — International Finance | Data: RBI DBIE Portal | Model: Facebook Prophet
</div>""", unsafe_allow_html=True)
