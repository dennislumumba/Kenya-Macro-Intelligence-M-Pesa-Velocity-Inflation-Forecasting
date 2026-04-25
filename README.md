# Economic Intelligence Hub — Kenya Macro-Fintech Forecasting Engine

A data-driven project analyzing the link between M-Pesa transaction velocity and inflation in Kenya.

The model explores how mobile money activity can act as a leading indicator for macroeconomic trends, applying time-series analysis and forecasting techniques to generate insights on inflation dynamics.

This project demonstrates the potential of alternative financial data in real-time economic intelligence and policy-relevant forecasting.

This repository answers one commercially relevant question: **how does M-Pesa transaction velocity interact with inflation and monetary policy in Kenya?**

It does that by combining four layers:

1. **Macro ingestion and cleaning** of a synthetic 5-year monthly Kenya dataset.
2. **SARIMA inflation forecasting** for the next 12 months.
3. **GARCH volatility modelling** on mobile money transaction flows.
4. **Boardroom interpretation** of how CBR moves may suppress or accelerate transaction velocity.

---

## Why this project matters in the Kenyan market

In Kenya, mobile money is not just a fintech metric. It is a live economic pulse. When transaction values slow, that can reflect consumer stress, tighter liquidity, weaker merchant turnover, or policy drag from high rates. When they accelerate, they often signal confidence, faster circulation, and stronger commercial activity.

That is why an economist who can connect **CBK policy**, **consumer inflation**, and **digital transaction behaviour** has a real strategic edge.

This project is deliberately tailored to Dennis Lumumba's profile:

- Kenyatta University training in **Economics and Statistics**
- Practical strength in **Python, dashboards, predictive analytics, and strategy**
- A positioning opportunity to become a specialist in **Kenyan economic intelligence** instead of a generic global data analyst

---

## Project structure

```text
.
├── forecasting_engine.py   # Core econometric + volatility engine
├── macro_dashboard.py      # Streamlit executive dashboard
└── README.md               # Strategic advisory note and operating guide
```

When `forecasting_engine.py` runs, it also creates an `artifacts/` folder with:

- `mock_cbk_raw_data.csv`
- `mock_cbk_cleaned_data.csv`
- `inflation_holdout_forecast.csv`
- `inflation_12m_forecast.csv`
- `historical_market_volatility.csv`
- `future_market_volatility.csv`
- `diagnostics.json`

---

## Analytical design

### 1) Data ingestion and cleaning
The engine creates a **realistic mock monthly dataset** for 60 months with:

- `cpi`
- `mobile_money_value_kes_bn`
- `cbr`

It then cleans the data by:

- enforcing monthly date order
- interpolating missing values
- recomputing growth rates
- clipping extreme synthetic outliers so volatility estimates remain interpretable

This is important because in real Kenya workflows, CBK and KNBS series are often released on different schedules, and operational analytics always starts with alignment before modelling.

### 2) Inflation forecasting with SARIMA
The inflation forecasting layer models **year-on-year inflation derived from CPI**.

Why this matters:

- executives think in inflation regimes, not raw price index levels
- YoY inflation is closer to how monetary policy discussions are framed in Kenya
- SARIMA captures persistence plus seasonality, which is useful in monthly macro series

The engine performs:

- holdout backtest on the latest 12 months
- MAE and RMSE tracking
- forward 12-month forecast with confidence bands

### 3) Mobile money volatility with GARCH
The GARCH layer models volatility in `mpesa_growth_pct`.

This matters because average growth alone is not enough. Boards want to know:

- Is the transaction system stable?
- Is velocity becoming more fragile?
- Is market stress rising even before top-line values break?

The model converts conditional volatility into a **Market Volatility Index (0–100)** so a non-technical executive can understand it immediately.

### 4) Statistical rigor
The engine includes two important diagnostics:

- **ADF tests** for stationarity
- **Granger causality tests** to assess whether mobile money growth has predictive content for inflation shifts

This is where the project stops being generic. It is not just showing a forecast. It is checking whether the underlying assumptions even justify time-series modelling.

### 5) Executive ROI summary
A simple policy pass-through regression estimates how **CBR changes relate to transaction velocity**.

This creates a board-level commercial readout:

- what a 100bps hike may do to velocity
- what a 100bps cut may unlock in additional monthly throughput
- how to translate policy into a merchant and fintech growth narrative

---

## Strategic interpretation of the current system

The engine is designed to make a few boardroom truths obvious:

1. **Inflation and digital money flows should not be analysed separately in Kenya.**
   That is intellectually lazy and commercially expensive.

2. **Volatility is often more informative than level.**
   A stable slowdown and a chaotic slowdown require very different decisions.

3. **Rate policy affects more than borrowing costs.**
   In a mobile-money-first economy, policy changes can propagate through transaction behaviour, merchant turnover, and household liquidity speed.

4. **This can become a recurring intelligence product.**
   Not just a GitHub project. A real advisory product for banks, fintechs, investment teams, and strategy leaders operating in East Africa.

---

## How to run

### Install the required packages

```bash
pip install pandas numpy statsmodels arch streamlit plotly
```

### Run the forecasting engine

```bash
python forecasting_engine.py
```

### Launch the dashboard

```bash
streamlit run macro_dashboard.py
```

---

## Dashboard layout

The Streamlit app is intentionally executive-facing, not academic.

It includes:

### Inflation Forecast vs Actuals
Shows historical YoY inflation, holdout forecast performance, and the next 12-month outlook.

### Market Volatility Index
Turns GARCH output into an interpretable fintech stress indicator.

### Executive ROI Summary
Translates rate changes into estimated transaction-velocity effects in plain commercial language.

### Diagnostics tables
Keeps the analysis honest with:

- ADF stationarity results
- Granger causality p-values
- cleaned macro dataset for inspection

---

Marketing as:

> “I build economic intelligence systems for African markets that connect policy, inflation, digital payments, and commercial decision-making.”

---

## Next evolution

the next version should add:

1. **Live data connectors** for CBK and KNBS releases
2. **Scenario analysis** for rate hike / rate cut / inflation shock cases
3. **Vector autoregression (VAR)** to model feedback loops across CPI, CBR, and transaction values
4. **Segment views** for banks, fintechs, FMCG, and investors
5. **Automated briefing exports** to PDF or email for executives every month


---

## Data notes and source anchors

The current repository uses a **synthetic mock dataset** for demonstration. However, its design mirrors the structure of official Kenyan macro sources:

- CBK Monthly Economic Indicators: https://www.centralbank.go.ke/monthly-economic-indicators/
- CBK Mobile Payments statistics: https://www.centralbank.go.ke/national-payments-system/mobile-payments/
- KNBS CPI and Inflation Rates: https://www.knbs.or.ke/cpi-and-inflation-rates/

These sources define the real-world framing of inflation, mobile money activity, and policy-rate context for a production version of this system.

---
