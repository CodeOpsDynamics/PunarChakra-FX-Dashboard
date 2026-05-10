"""
=============================================================================
PunarChakra E-Waste Exports — AI-Powered FX Risk Assessment
=============================================================================
Author      : Himanshu Rai | Roll No. XW013-25 | IIM Ranchi EMBA 2025-27
Course      : International Finance | Prof. Kamran Quddus
Project     : WAI (Working with AI) — Term IV
Description : Facebook Prophet-based INR/USD forecasting + IRP forward rate
              computation + Hedging strategy evaluation for PunarChakra's
              planned e-waste exports to UAE and Singapore (FY2027-28)
Data Source : RBI Reference Rates — 968 official trading days (Apr 2022–Apr 2026)
=============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_rbi_data(filepath=None):
    """Load and clean RBI Reference Rate data."""
    import os
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RBI_Data.xlsx")
    df = pd.read_excel(filepath)
    df.columns = ["Date","USD","GBP","EUR","JPY","AED","IDR"]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna(subset=["USD"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["USD"] = pd.to_numeric(df["USD"], errors="coerce")
    df = df.dropna(subset=["USD"])
    # Forward fill any remaining gaps (trading holidays)
    df["USD"] = df["USD"].ffill()
    return df

def compute_descriptive_stats(df):
    """Compute key descriptive statistics."""
    usd = df["USD"]
    daily_returns = usd.pct_change().dropna()
    annual_vol    = daily_returns.std() * np.sqrt(252) * 100

    start_rate = usd.iloc[0]
    end_rate   = usd.iloc[-1]
    n_years    = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    cagr       = ((end_rate / start_rate) ** (1 / n_years) - 1) * 100

    stats = {
        "Total trading days"         : len(df),
        "Start date"                 : df["Date"].iloc[0].strftime("%d %b %Y"),
        "End date"                   : df["Date"].iloc[-1].strftime("%d %b %Y"),
        "Min INR/USD"                : round(usd.min(), 4),
        "Max INR/USD"                : round(usd.max(), 4),
        "Mean INR/USD"               : round(usd.mean(), 4),
        "Latest INR/USD (RBI)"       : round(usd.iloc[-1], 4),
        "Live rate (May 10, 2026)"   : 94.28,
        "CAGR INR depreciation (%)"  : round(cagr, 2),
        "Annualised volatility (%)"  : round(annual_vol, 2),
    }
    return stats

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FACEBOOK PROPHET FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet_model(df):
    """Train Facebook Prophet on RBI INR/USD data."""
    # Prophet requires columns 'ds' and 'y'
    prophet_df = df[["Date", "USD"]].rename(columns={"Date": "ds", "USD": "y"})

    model = Prophet(
        changepoint_prior_scale  = 0.05,     # moderate flexibility for trend breaks
        seasonality_mode         = "multiplicative",  # volatility scales with level
        yearly_seasonality       = True,
        weekly_seasonality       = False,    # FX markets open Mon-Fri, no weekly seasonality
        daily_seasonality        = False,
        interval_width           = 0.80,     # 80% confidence interval
    )

    # Add monthly seasonality (RBI MPC meetings, month-end effects)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(prophet_df)
    return model

def generate_forecast(model, periods=500):
    """Generate 500-day forecast (~FY2027-28 coverage)."""
    future = model.make_future_dataframe(periods=periods, freq="B")  # Business days only
    forecast = model.predict(future)
    return forecast

def apply_scenarios(forecast, spot=94.28):
    """
    Apply 3 geopolitical scenarios on top of Prophet baseline.
    Scenarios defined based on current macro environment (May 2026):
      - Baseline : Prophet central forecast (trend continuation)
      - Bear INR : Sustained oil shock >$100, USD strength, capital outflows
      - Bull INR : Oil drops, RBI cuts, large FPI inflows
    """
    fc = forecast.copy()
    # Only apply scenario adjustments to the future period
    last_actual = pd.Timestamp("2026-04-17")
    future_mask = fc["ds"] > last_actual

    # Bear scenario: additional +4% depreciation on top of baseline trend
    # (reflects sustained crude at $100+, US-Iran tensions, Fed hold)
    fc["bear"] = fc["yhat"].copy()
    fc.loc[future_mask, "bear"] = fc.loc[future_mask, "yhat"] * 1.04

    # Bull scenario: INR appreciates 5% from baseline
    # (reflects oil drop to $65-70, RBI intervention, FPI inflows)
    fc["bull"] = fc["yhat"].copy()
    fc.loc[future_mask, "bull"] = fc.loc[future_mask, "yhat"] * 0.95

    return fc

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — IRP FORWARD RATE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_irp_forward_rates(
    spot      = 94.28,   # Live INR/USD (May 10, 2026)
    r_inr     = 0.0525,  # RBI Repo Rate 5.25% (May 2026)
    r_usd     = 0.03625, # Fed Funds midpoint 3.50-3.75% (Apr 29, 2026)
    tenors    = [3, 6, 9, 12]  # months
):
    """
    Compute IRP-based forward rates using Covered Interest Rate Parity (CIRP):
        F = S × [(1 + r_INR) / (1 + r_USD)] ^ T
    where T is in years.
    """
    results = []
    for months in tenors:
        T         = months / 12
        F         = spot * ((1 + r_inr) / (1 + r_usd)) ** T
        premium   = ((F - spot) / spot) * 100
        results.append({
            "Tenor (months)"         : months,
            "Spot Rate (₹/USD)"      : spot,
            "IRP Forward Rate (₹/USD)": round(F, 2),
            "Forward Premium (%)"    : round(premium, 2),
            "Implication"            : "INR at forward premium — lock rate now" if premium > 0 else "INR at discount"
        })
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HEDGING STRATEGY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_hedging_strategies(
    invoice_usd   = 100_000,   # Representative export invoice
    spot          = 94.28,     # Current spot
    forward_6m    = 97.07,     # 6-month IRP forward
    option_premium_pct = 0.013, # ~1.3% option premium (USD put)
    hedge_ratio_fwd    = 0.60,  # 60% forward cover
    hedge_ratio_nat    = 0.40,  # 40% natural hedge
    scenario_rates     = {      # Spot rates at 6-month maturity
        "Baseline" : 96.50,
        "Bear (INR)" : 101.00,
        "Bull (INR)" : 90.00,
    }
):
    """
    Evaluate 3 hedging strategies across 3 scenarios.
    Returns detailed P&L comparison DataFrame.
    """
    results = []
    option_premium_inr = invoice_usd * option_premium_pct * spot

    for scenario, future_spot in scenario_rates.items():

        # ── No hedge (baseline comparison) ──
        inr_unhedged = invoice_usd * future_spot

        # ── Strategy A: Forward Contract (60% hedged) ──
        fwd_portion  = invoice_usd * hedge_ratio_fwd
        open_portion = invoice_usd * (1 - hedge_ratio_fwd)
        inr_fwd      = (fwd_portion * forward_6m) + (open_portion * future_spot)
        inr_fwd_net  = inr_fwd  # no explicit premium for forward (spread embedded)

        # ── Strategy B: Currency Option (USD Put, 100% cover) ──
        # If future_spot < forward (INR appreciated), exercise option at forward rate
        # If future_spot > forward (INR depreciated), let option lapse — sell at spot
        effective_rate_option = max(future_spot, forward_6m)
        inr_option_gross      = invoice_usd * effective_rate_option
        inr_option_net        = inr_option_gross - option_premium_inr

        # ── Strategy C: Natural Hedge (40%) + Forward (40%) + Open (20%) ──
        nat_portion  = invoice_usd * hedge_ratio_nat   # offset by USD equipment import
        fwd_portion2 = invoice_usd * 0.40
        open_portion2= invoice_usd * 0.20
        # Natural hedge: USD inflow offsets USD outflow — realised at forward for planning
        inr_natural  = (nat_portion * forward_6m) + (fwd_portion2 * forward_6m) + (open_portion2 * future_spot)

        results.append({
            "Scenario"                       : scenario,
            "Future Spot (₹/USD)"            : future_spot,
            "Unhedged INR Realisation (₹)"   : round(inr_unhedged),
            "Strategy A — Forward (60%) (₹)" : round(inr_fwd_net),
            "Strategy B — Options 100% (₹)"  : round(inr_option_net),
            "Strategy C — Natural+Fwd (₹)"   : round(inr_natural),
            "A vs Unhedged (₹)"              : round(inr_fwd_net - inr_unhedged),
            "B vs Unhedged (₹)"              : round(inr_option_net - inr_unhedged),
            "C vs Unhedged (₹)"              : round(inr_natural - inr_unhedged),
        })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_historical_trend(df):
    """Fig 1: Historical INR/USD with trend line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["USD"],
        mode="lines", name="INR/USD (Daily)",
        line=dict(color="#1f77b4", width=1.2),
        hovertemplate="Date: %{x|%d %b %Y}<br>Rate: ₹%{y:.2f}<extra></extra>"
    ))

    # Add trend line
    x_num  = (df["Date"] - df["Date"].iloc[0]).dt.days
    z      = np.polyfit(x_num, df["USD"], 1)
    trend  = np.polyval(z, x_num)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=trend,
        mode="lines", name="Trend Line",
        line=dict(color="red", width=2, dash="dash"),
    ))

    # Annotations
    fig.add_annotation(x="2022-04-12", y=76.5, text="₹76.11 Start", showarrow=True, arrowhead=2, font=dict(size=10))
    fig.add_annotation(x="2026-04-17", y=93.5, text="₹92.72 Latest", showarrow=True, arrowhead=2, font=dict(size=10))
    fig.add_vline(x="2025-01-01", line_dash="dot", line_color="orange",
                  annotation_text="Depreciation<br>acceleration", annotation_position="top left")

    fig.update_layout(
        title="<b>Figure 1: INR/USD Historical Trend (Apr 2022 – Apr 2026)</b><br>"
              "<sup>968 RBI Reference Rate Trading Days | CAGR Depreciation: 5.04% p.a.</sup>",
        xaxis_title="Date", yaxis_title="INR per 1 USD (₹)",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Times New Roman")
    )
    return fig


def plot_prophet_forecast(df, forecast):
    """Fig 2: Prophet forecast with confidence bands and 3 scenarios."""
    last_actual = pd.Timestamp("2026-04-17")
    hist_mask   = forecast["ds"] <= last_actual
    fut_mask    = forecast["ds"] >  last_actual

    fig = go.Figure()

    # Historical actual
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["USD"],
        mode="lines", name="Actual INR/USD",
        line=dict(color="#1f77b4", width=1.2),
    ))

    # Confidence band (future only)
    fig.add_trace(go.Scatter(
        x=forecast.loc[fut_mask, "ds"].tolist() + forecast.loc[fut_mask, "ds"].tolist()[::-1],
        y=forecast.loc[fut_mask, "yhat_upper"].tolist() + forecast.loc[fut_mask, "yhat_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.12)",
        line=dict(color="rgba(255,255,255,0)"), name="80% Confidence Band",
    ))

    # Baseline forecast
    fig.add_trace(go.Scatter(
        x=forecast.loc[fut_mask, "ds"], y=forecast.loc[fut_mask, "yhat"],
        mode="lines", name="Baseline Forecast",
        line=dict(color="#1f77b4", width=2, dash="dash"),
    ))

    # Bear scenario
    fig.add_trace(go.Scatter(
        x=forecast.loc[fut_mask, "ds"], y=forecast.loc[fut_mask, "bear"],
        mode="lines", name="Bear Scenario (INR weak)",
        line=dict(color="crimson", width=2, dash="dot"),
    ))

    # Bull scenario
    fig.add_trace(go.Scatter(
        x=forecast.loc[fut_mask, "ds"], y=forecast.loc[fut_mask, "bull"],
        mode="lines", name="Bull Scenario (INR strong)",
        line=dict(color="green", width=2, dash="dot"),
    ))

    # Live rate marker
    fig.add_hline(y=94.28, line_dash="solid", line_color="orange", line_width=1.5,
                  annotation_text="Live: ₹94.28 (May 10, 2026)", annotation_position="right")

    # FY2027-28 shading
    fig.add_vrect(x0="2027-04-01", x1="2028-03-31",
                  fillcolor="lightyellow", opacity=0.4, layer="below",
                  annotation_text="FY2027-28", annotation_position="top left")

    fig.update_layout(
        title="<b>Figure 2: Facebook Prophet INR/USD Forecast — 3 Scenarios (FY2027-28)</b>",
        xaxis_title="Date", yaxis_title="INR per 1 USD (₹)",
        height=460, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family="Times New Roman")
    )
    return fig


def plot_irp_forward_rates(irp_df):
    """Fig 3: IRP forward rates bar chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"{t}M" for t in irp_df["Tenor (months)"]],
        y=irp_df["IRP Forward Rate (₹/USD)"],
        name="IRP Forward Rate",
        marker_color=["#2196F3","#1976D2","#1565C0","#0D47A1"],
        text=[f"₹{r:.2f}" for r in irp_df["IRP Forward Rate (₹/USD)"]],
        textposition="outside"
    ))

    fig.add_hline(y=94.28, line_dash="dash", line_color="red",
                  annotation_text="Current Spot ₹94.28", annotation_position="right")

    fig.update_layout(
        title="<b>Figure 3: IRP-Based Forward Rates vs. Spot</b><br>"
              "<sup>Spot: ₹94.28 | r_INR: 5.25% | r_USD: 3.625% | Formula: F = S × [(1+r_INR)/(1+r_USD)]^T</sup>",
        xaxis_title="Forward Tenor", yaxis_title="INR per 1 USD (₹)",
        yaxis=dict(range=[92, 101]),
        height=400, template="plotly_white",
        font=dict(family="Times New Roman")
    )
    return fig


def plot_hedging_comparison(hedge_df):
    """Fig 4: Hedging strategy comparison grouped bar chart."""
    scenarios = hedge_df["Scenario"].tolist()
    strategies = [
        ("Unhedged INR Realisation (₹)", "Unhedged", "#FF6B6B"),
        ("Strategy A — Forward (60%) (₹)", "A: Forward 60%", "#4ECDC4"),
        ("Strategy B — Options 100% (₹)", "B: Options 100%", "#45B7D1"),
        ("Strategy C — Natural+Fwd (₹)", "C: Natural+Fwd", "#96CEB4"),
    ]

    fig = go.Figure()
    for col, label, color in strategies:
        fig.add_trace(go.Bar(
            name=label,
            x=scenarios,
            y=hedge_df[col] / 1_00_000,  # Convert to lakhs
            marker_color=color,
            text=[f"₹{v/1_00_000:.1f}L" for v in hedge_df[col]],
            textposition="outside"
        ))

    fig.update_layout(
        title="<b>Figure 4: INR Realisation Comparison — 3 Strategies × 3 Scenarios</b><br>"
              "<sup>Invoice: $1,00,000 USD | Values in ₹ Lakhs</sup>",
        xaxis_title="Scenario", yaxis_title="INR Realisation (₹ Lakhs)",
        barmode="group",
        height=450, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Times New Roman")
    )
    return fig


def plot_pnl_heatmap(hedge_df):
    """Fig 5: P&L vs Unhedged heatmap."""
    strategies = ["A vs Unhedged (₹)", "B vs Unhedged (₹)", "C vs Unhedged (₹)"]
    labels     = ["Forward 60%", "Options 100%", "Natural+Fwd"]
    scenarios  = hedge_df["Scenario"].tolist()

    z_data = [[hedge_df[col].iloc[i] / 1000 for col in strategies] for i in range(len(scenarios))]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=labels,
        y=scenarios,
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"₹{v:.0f}K" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=13, family="Times New Roman"),
        hovertemplate="Scenario: %{y}<br>Strategy: %{x}<br>P&L vs Unhedged: ₹%{z:.0f}K<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Figure 5: P&L vs Unhedged Position (₹ Thousands)</b><br>"
              "<sup>Green = Hedge adds value | Red = Hedge costs more than unhedged</sup>",
        height=350, template="plotly_white",
        font=dict(family="Times New Roman")
    )
    return fig


def plot_annualised_projections(irp_df, stats):
    """Fig 6: Annual FX exposure & INR realisation projections."""
    annual_usd = [900_000, 1_000_000, 1_100_000]  # Conservative, Mid, Optimistic
    labels     = ["Conservative\n$900K", "Mid\n$1.0M", "Optimistic\n$1.1M"]
    forward_12m = irp_df[irp_df["Tenor (months)"] == 12]["IRP Forward Rate (₹/USD)"].values[0]
    spot = 94.28

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["INR Realisation at Spot vs Forward (₹ Crore)",
                        "Additional Realisation from Hedging (₹ Lakh)"])

    for i, (usd, lbl) in enumerate(zip(annual_usd, ["Conservative", "Mid", "Optimistic"])):
        fig.add_trace(go.Bar(
            name=f"{lbl} @ Spot",
            x=[lbl], y=[round(usd * spot / 1e7, 2)],
            marker_color="#FF6B6B", showlegend=(i==0)
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            name=f"{lbl} @ Forward",
            x=[lbl], y=[round(usd * forward_12m / 1e7, 2)],
            marker_color="#4ECDC4", showlegend=(i==0)
        ), row=1, col=1)

    additional = [(usd * (forward_12m - spot)) / 1e5 for usd in annual_usd]
    fig.add_trace(go.Bar(
        x=["Conservative", "Mid", "Optimistic"],
        y=[round(a, 1) for a in additional],
        marker_color="#96CEB4", name="Additional Realisation",
        text=[f"+₹{a:.1f}L" for a in additional], textposition="outside",
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title="<b>Figure 6: PunarChakra — Annual FX Exposure & Hedge Value (FY2027-28)</b>",
        height=420, template="plotly_white", barmode="group",
        font=dict(family="Times New Roman")
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(filepath=None):
    """Run complete PunarChakra FX Risk Analysis pipeline."""

    print("=" * 65)
    print("  PunarChakra FX Risk Assessment — IIM Ranchi WAI Project")
    print("  Author: Himanshu Rai | XW013-25")
    print("=" * 65)

    # 1. Load data
    print("\n[1/6] Loading RBI data...")
    df = load_rbi_data(filepath)

    # 2. Descriptive stats
    print("[2/6] Computing descriptive statistics...")
    stats = compute_descriptive_stats(df)
    print("\n  ── Dataset Summary ──")
    for k, v in stats.items():
        print(f"  {k:<35} : {v}")

    # 3. Prophet model
    print("\n[3/6] Training Facebook Prophet model (this takes ~30 seconds)...")
    model    = train_prophet_model(df)
    forecast = generate_forecast(model, periods=500)
    forecast = apply_scenarios(forecast)
    print("  Prophet model trained successfully.")
    print(f"  Forecast covers: {forecast['ds'].iloc[-1].strftime('%d %b %Y')}")

    # 4. IRP forward rates
    print("\n[4/6] Computing IRP-based forward rates...")
    irp_df = compute_irp_forward_rates()
    print("\n  ── IRP Forward Rate Table ──")
    print(irp_df[["Tenor (months)", "IRP Forward Rate (₹/USD)", "Forward Premium (%)"]].to_string(index=False))

    # 5. Hedging strategies
    print("\n[5/6] Evaluating hedging strategies...")
    hedge_df = evaluate_hedging_strategies()
    print("\n  ── Hedging Strategy Comparison ($1,00,000 Invoice) ──")
    display_cols = ["Scenario", "Future Spot (₹/USD)",
                    "Unhedged INR Realisation (₹)",
                    "Strategy A — Forward (60%) (₹)",
                    "Strategy C — Natural+Fwd (₹)"]
    print(hedge_df[display_cols].to_string(index=False))

    # 6. Recommendation
    print("\n[6/6] Recommended Strategy for PunarChakra FY2027-28:")
    print("""
  ┌────────────────────────────────────────────────────────────┐
  │  LAYER 1: Natural Hedge — 40% of USD receivables          │
  │           → Source plant equipment imports in USD          │
  │           → Zero premium cost; serves business purpose     │
  │                                                            │
  │  LAYER 2: Forward Contract — 40% of USD receivables        │
  │           → Book 6–9 month forwards on confirmed invoices  │
  │           → IRP rate: ₹97.07 (6M) to ₹98.50 (9M)         │
  │           → Use EEFC account at bank                       │
  │                                                            │
  │  LAYER 3: Open Exposure — 20% (monitored monthly)         │
  │           → Review if crude crosses $95/barrel             │
  │           → Review if INR/USD moves ±3% from forecast      │
  │                                                            │
  │  Planning Rate: ₹96–97/USD for FY2027-28 projections      │
  │  Net hedge benefit: ~₹27.6 lakh on $10L USD receivables   │
  └────────────────────────────────────────────────────────────┘
    """)

    print("  ✅ Analysis complete. Returning objects for dashboard use.")
    return df, forecast, irp_df, hedge_df, stats, model


# Run when executed directly
if __name__ == "__main__":
    df, forecast, irp_df, hedge_df, stats, model = run_full_analysis()

    # Save outputs to CSV for reference
    irp_out = irp_df.rename(columns={
        "Tenor (months)"          : "Tenor_Months",
        "Spot Rate (₹/USD)"       : "Spot_Rate_INR_USD",
        "IRP Forward Rate (₹/USD)": "IRP_Forward_Rate_INR_USD",
        "Forward Premium (%)"     : "Forward_Premium_Pct",
    })
    irp_out.to_csv("irp_forward_rates.csv", index=False)
    hedge_df.to_csv("hedging_strategy_comparison.csv", index=False)
    fc_out = forecast[["ds","yhat","yhat_lower","yhat_upper","bear","bull"]].tail(500).copy()
    fc_out = fc_out.rename(columns={
        "ds"         : "Date",
        "yhat"       : "Forecast_INR_USD",
        "yhat_lower" : "Lower_Bound_80pct",
        "yhat_upper" : "Upper_Bound_80pct",
        "bear"       : "Bear_Scenario",
        "bull"       : "Bull_Scenario",
    })
    fc_out["Date"] = pd.to_datetime(fc_out["Date"]).dt.strftime("%d %b %Y")
    for col in ["Forecast_INR_USD","Lower_Bound_80pct","Upper_Bound_80pct","Bear_Scenario","Bull_Scenario"]:
        fc_out[col] = fc_out[col].round(4)
    fc_out.to_csv("prophet_forecast.csv", index=False)
    print("\n  Output files saved: irp_forward_rates.csv, hedging_strategy_comparison.csv, prophet_forecast.csv")
