# ♻️ PunarChakra — AI-Powered FX Risk Assessment Dashboard

> **Hedging INR/USD Exposure for Indian E-Waste Exports to UAE & Singapore (FY2027–28)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://punarchakra-fx-dashboard.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Prophet](https://img.shields.io/badge/AI%20Model-Facebook%20Prophet-orange)
![License](https://img.shields.io/badge/License-Academic%20Use-green)

---

## 📌 About This Project

**PunarChakra** is a pre-incorporated e-waste recycling venture being established in **Ghazipur, Uttar Pradesh**, with factory setup planned for FY2026–27 and first export shipments targeting FY2027–28. The venture processes end-of-life electronics — recovering copper, aluminium, gold, and silver — and exports refined materials to buyers in **UAE (Sharjah)** and **Singapore (TES-AMM network)**.

Since all export revenues are denominated in **USD** while all operational costs are in **INR**, PunarChakra carries a structural foreign exchange (FX) risk from Day 1. As of May 2026, INR/USD stands at ₹94.28 — having hit a record low of ₹95.2 driven by US–Iran tensions and Brent crude above $101/barrel.

This project uses **Facebook Prophet (AI forecasting)** trained on **968 official RBI Reference Rate trading days (April 2022 – April 2026)** to forecast INR/USD under three geopolitical scenarios, compute IRP-based forward rates, and evaluate three hedging strategies — recommending the optimal approach for a pre-revenue, capital-constrained startup.

---

## 🎓 Academic Context

| Field | Detail |
|---|---|
| **Institution** | Indian Institute of Management Ranchi |
| **Programme** | Executive MBA — Winter Batch 2025–27 |
| **Term** | IV — Elective |
| **Course** | International Finance |
| **Faculty** | Prof. Kamran Quddus |
| **Project Type** | Working with AI (WAI) — Individual Project |
| **Submitted by** | Himanshu Rai \| Roll No. XW013-25 |

---

## 🚀 Live Dashboard

> 🔗 **[Open Live Dashboard →](https://punarchakra-fx-dashboard.streamlit.app)**

The interactive Streamlit dashboard allows you to:
- Adjust **annual USD receivables**, **hedge ratios**, and **interest rates** via sidebar sliders
- Toggle between **3 geopolitical scenarios** (Baseline / Bear / Bull)
- View **6 interactive charts** including Prophet forecast, IRP rates, and hedging P&L comparison
- See a **real-time recommendation** that updates as you change inputs

---

## 📊 Key Findings

| Metric | Value |
|---|---|
| INR CAGR Depreciation (2022–26) | **5.04% per annum** |
| Annualised Volatility | **3.99%** |
| Current Spot Rate (May 2026) | **₹94.28/USD** |
| 6-Month IRP Forward Rate | **₹95.02/USD** |
| 12-Month IRP Forward Rate | **₹95.76/USD** |
| Baseline Forecast FY2027–28 | **₹96–99/USD** |
| Bear Scenario (crude >$100) | **₹101–104/USD** |
| Net Hedge Benefit (Year 1) | **~₹6.5 Lakh on $1M exports** |

### Recommended Strategy

```
Layer 1 (40%) → Natural Hedge    — Source plant equipment in USD (zero premium)
Layer 2 (40%) → Forward Contract — Book 6–9M forwards on confirmed invoices
Layer 3 (20%) → Open Exposure    — Monitor monthly via Prophet dashboard
```

---

## 🧠 AI & Tech Stack

| Tool | Purpose |
|---|---|
| **Facebook Prophet** | INR/USD time-series forecasting with changepoints, seasonality, 80% CI |
| **Claude (Anthropic)** | Literature synthesis, methodology design, report drafting |
| **Python** (pandas, numpy, plotly) | Data processing, IRP computation, visualisations |
| **Streamlit** | Interactive dashboard with live sliders and scenario toggles |

---

## 📁 Repository Structure

```
PunarChakra-FX-Dashboard/
│
├── punarchakra_dashboard.py    # Streamlit interactive dashboard (main app)
├── punarchakra_analysis.py     # Standalone analysis script (Prophet + IRP + Hedging)
├── RBI_Data.xlsx               # 968 RBI Reference Rate trading days (Apr 2022–Apr 2026)
├── requirements.txt            # Python dependencies
│
├── .streamlit/
│   └── config.toml             # Streamlit theme and server config
│
└── README.md                   # This file
```

---

## ⚙️ Run Locally

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/PunarChakra-FX-Dashboard.git
cd PunarChakra-FX-Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run punarchakra_dashboard.py
```

The app will open at `http://localhost:8501`

### Run standalone analysis (terminal output)

```bash
python punarchakra_analysis.py
```

This will print descriptive stats, IRP forward rate table, hedging comparison, and save 3 CSV output files.

---

## 📐 Methodology

### 1. Data
- **Source:** RBI DBIE portal (official Reference Rates)
- **Period:** April 12, 2022 – April 17, 2026
- **Size:** 968 business-day observations
- **Preprocessing:** Missing values forward-filled for trading holidays

### 2. Facebook Prophet Model
```
Changepoint prior scale  : 0.05  (moderate — detects regime shifts without overfitting)
Seasonality mode         : Multiplicative  (volatility scales with rate level)
Yearly seasonality       : Enabled
Monthly seasonality      : Custom Fourier order 5
Confidence interval      : 80%
Validation               : Walk-forward cross-validation, 180-day horizon
Forecast horizon         : 24 months (April 2026 – March 2028)
```

### 3. Interest Rate Parity (Covered IRP)
```
F = S × [(1 + r_INR) / (1 + r_USD)] ^ T

Where:
  S      = ₹94.28  (spot rate, May 2026)
  r_INR  = 5.25%   (RBI Repo Rate, Feb 2026)
  r_USD  = 3.625%  (Fed Funds midpoint, Apr 2026)
  T      = tenor in years (0.25 / 0.5 / 0.75 / 1.0)
```

### 4. Geopolitical Scenarios
| Scenario | Assumption | FY2027–28 Projection |
|---|---|---|
| Baseline | Crude $80–90; Fed cuts 1×; RBI stable | ₹96–99 |
| Bear (INR weak) | Crude >$100; US-Iran escalation; Fed holds | ₹101–104 |
| Bull (INR strong) | Oil drops; RBI intervention; FPI inflows | ₹88–91 |

---

## ⚖️ Ethical Considerations

- **Data sources:** All data from official public sources — RBI DBIE and Federal Reserve FRED. No personal or proprietary data used.
- **Model bias:** Prophet trained on 2022–26 (depreciation-heavy period) may underestimate INR appreciation. Bull scenario explicitly stress-tests the opposite.
- **Not financial advice:** Dashboard outputs are internal planning tools only. All derivative contracts must be booked with an RBI-regulated forex dealer after professional consultation.
- **AI use disclosure:** Claude (Anthropic) was used for synthesis and drafting only. All financial figures were independently computed in Python and verified against official data.

---

## 📚 Key References

1. Allayannis, G., & Weston, J. P. (2001). The use of foreign currency derivatives and firm market value. *Review of Financial Studies*, 14(1), 243–276.
2. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *PeerJ Computer Science*, 4, e164.
3. Misra, S., & Behera, H. (2006). Non-deliverable foreign exchange forward market. *RBI Occasional Papers*, 27(3).
4. Balde, C.P. et al. (2024). *Global E-Waste Monitor 2024*. United Nations University.
5. Reserve Bank of India. (2026). RBI Reference Rates. https://dbie.rbi.org.in

---

## 📬 Contact

**Himanshu Rai**
Executive MBA 2025–27 (Winter Batch) | IIM Ranchi
Roll No. XW013-25 | Term IV

---

*This project is submitted as part of the Working with AI (WAI) assessment for International Finance (Term IV), IIM Ranchi EMBA 2025–27. For academic use only.*
