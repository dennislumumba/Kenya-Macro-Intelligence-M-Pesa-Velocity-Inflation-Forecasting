from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting_engine import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="Economic Intelligence Hub | Kenya",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_outputs() -> dict:
    return run_pipeline(output_dir=ARTIFACT_DIR)


outputs = load_outputs()
cleaned = outputs["cleaned_data"].copy()
sarima = outputs["sarima"]
garch = outputs["garch"]
roi = outputs["roi_summary"]
stationarity = outputs["stationarity"]
granger = outputs["granger_causality"]
note = outputs["strategic_note"]

historical_actuals = sarima["historical_actuals"].copy()
holdout = sarima["holdout_forecast"].copy()
future_forecast = sarima["future_forecast"].copy()
historical_vol = garch["historical_volatility"].copy()
future_vol = garch["future_volatility"].copy()

for df in [cleaned, historical_actuals, holdout, future_forecast, historical_vol, future_vol]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])


st.title("Economic Intelligence Hub: Kenya Macro-Fintech Command Centre")
st.caption(
    "Designed for Dennis Lumumba — economist, data strategist, and founder — to turn Kenyan macro data into boardroom-grade decisions."
)

with st.sidebar:
    st.header("Why this matters")
    st.write(
        "This dashboard treats M-Pesa velocity as more than a payments metric. In Kenya, it is a live signal of consumer liquidity, merchant activity, and the speed at which policy changes travel into the real economy."
    )
    st.subheader("Analytical stack")
    st.write(
        "SARIMA forecasts inflation, GARCH quantifies fintech volatility, ADF tests validate time-series behaviour, and Granger causality checks whether transaction momentum predicts price pressure."
    )
    st.subheader("Data caveat")
    st.info(
        "The current app uses a realistic mock dataset shaped like monthly CBK-style indicators. Swap in real CBK/KNBS extracts when you are ready to operationalise it."
    )

latest_inflation = float(cleaned["inflation_yoy_pct"].dropna().iloc[-1])
forecast_avg = float(future_forecast["forecast_inflation_yoy_pct"].mean())
market_vol_idx = float(garch["current_market_volatility_index"])
monthly_uplift = float(roi["estimated_monthly_transaction_uplift_kes_bn_after_100bps_cut"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest YoY Inflation", f"{latest_inflation:.2f}%")
col2.metric("12M Avg Inflation Forecast", f"{forecast_avg:.2f}%")
col3.metric("Market Volatility Index", f"{market_vol_idx:.1f}", garch["volatility_regime"])
col4.metric("KES bn uplift after 100bps cut", f"{monthly_uplift:.2f}")

left, right = st.columns((2, 1))

with left:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical_actuals["date"],
            y=historical_actuals["actual_inflation_yoy_pct"],
            mode="lines",
            name="Actual inflation",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=holdout["date"],
            y=holdout["forecast_inflation_yoy_pct"],
            mode="lines",
            name="Backtest forecast",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_forecast["date"],
            y=future_forecast["forecast_inflation_yoy_pct"],
            mode="lines+markers",
            name="12M forward forecast",
            line=dict(color="#2ca02c", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_forecast["date"]) + list(future_forecast["date"][::-1]),
            y=list(future_forecast["upper_ci"]) + list(future_forecast["lower_ci"][::-1]),
            fill="toself",
            fillcolor="rgba(44,160,44,0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Future confidence band",
        )
    )
    fig.update_layout(
        title="Inflation Forecast vs Actuals",
        xaxis_title="Month",
        yaxis_title="Inflation YoY (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=market_vol_idx,
            title={"text": "Market Volatility Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7f3c8d"},
                "steps": [
                    {"range": [0, 45], "color": "#d9f0d3"},
                    {"range": [45, 67], "color": "#fee08b"},
                    {"range": [67, 100], "color": "#f46d43"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": market_vol_idx},
            },
        )
    )
    gauge.update_layout(template="plotly_white", height=300, margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("Executive readout")
    st.write(note["policy_readout"])
    st.write(note["commercial_implication"])
    st.write(note["decision_hook"])

vol_left, vol_right = st.columns((2, 1))

with vol_left:
    vol_fig = go.Figure()
    vol_fig.add_trace(
        go.Bar(
            x=historical_vol["date"],
            y=historical_vol["market_volatility_index"],
            name="Historical volatility index",
            marker_color="#7f3c8d",
            opacity=0.8,
        )
    )
    vol_fig.add_trace(
        go.Scatter(
            x=future_vol["date"],
            y=future_vol["forecast_market_volatility_index"],
            mode="lines+markers",
            name="Forward volatility signal",
            line=dict(color="#11a579", width=3),
        )
    )
    vol_fig.update_layout(
        title="Kenyan Fintech Market Volatility Index",
        xaxis_title="Month",
        yaxis_title="Index (0-100)",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(vol_fig, use_container_width=True)

with vol_right:
    st.subheader("Executive ROI Summary")
    st.markdown(
        f"""
**CBR beta to transaction velocity:** {roi['beta_cbr_change_to_velocity_pct_points']:.3f} percentage points  
**Impact of 100bps hike:** {roi['impact_of_100bps_hike_on_velocity_pct_points']:.3f} pp  
**Impact of 100bps cut:** {roi['impact_of_100bps_cut_on_velocity_pct_points']:.3f} pp  
**Monthly throughput uplift after cut:** KES {roi['estimated_monthly_transaction_uplift_kes_bn_after_100bps_cut']:.2f} bn  
**Annualised throughput uplift:** KES {roi['estimated_annual_transaction_uplift_kes_bn_after_100bps_cut']:.2f} bn
        """
    )
    st.info(roi["boardroom_message"])

st.divider()

mid1, mid2 = st.columns(2)

with mid1:
    st.subheader("ADF stationarity tests")
    adf_df = pd.DataFrame(stationarity).T.reset_index(drop=True)
    st.dataframe(adf_df, use_container_width=True)

with mid2:
    st.subheader("Granger causality verdict")
    st.write(granger["test"])
    st.write(
        f"Best lag: {granger['best_lag']} | Minimum p-value: {granger['minimum_p_value']:.4f} | Causality detected: {granger['granger_causality_detected']}"
    )
    granger_df = pd.DataFrame(granger["lag_results"]).T.reset_index().rename(columns={"index": "lag"})
    st.dataframe(granger_df, use_container_width=True)

with st.expander("Inspect cleaned mock macro dataset"):
    st.dataframe(cleaned, use_container_width=True)

st.markdown(
    """
### Operating guidance for Dennis
This is not a toy dashboard. It is the skeleton of an advisory product. Once connected to live CBK and KNBS extracts, it can become a recurring executive briefing engine for banks, fintechs, investment committees, and strategy teams tracking the Kenyan consumer economy.
"""
)
