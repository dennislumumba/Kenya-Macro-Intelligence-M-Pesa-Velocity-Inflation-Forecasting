from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}. Run forecasting_engine.py first.")
    return pd.read_csv(path, parse_dates=["date"])


def _safe_float(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_scenario_analysis(artifact_dir: str | Path = ARTIFACT_DIR) -> Dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    cleaned = _load_csv(artifact_dir / "mock_cbk_cleaned_data.csv")
    future_inflation = _load_csv(artifact_dir / "inflation_12m_forecast.csv")
    future_volatility = _load_csv(artifact_dir / "future_market_volatility.csv")

    sample = cleaned[["date", "cbr_change_pp", "mpesa_growth_pct", "mobile_money_value_kes_bn", "cbr"]].dropna().copy()
    if sample.empty:
        raise ValueError("Cleaned data does not contain enough observations to run scenario analysis.")

    last_observed = sample.iloc[-1]
    baseline_velocity = float(sample["mpesa_growth_pct"].tail(6).mean())
    latest_transaction_value = float(last_observed["mobile_money_value_kes_bn"])
    latest_cbr = float(last_observed["cbr"])
    avg_forward_inflation = float(future_inflation["forecast_inflation_yoy_pct"].mean())
    avg_forward_vol_index = float(future_volatility["forecast_market_volatility_index"].mean())

    cbr_change = sample["cbr_change_pp"].to_numpy(dtype=float)
    velocity = sample["mpesa_growth_pct"].to_numpy(dtype=float)

    if np.std(cbr_change) == 0:
        beta_cbr_to_velocity = -0.65
    else:
        beta_cbr_to_velocity = float(np.polyfit(cbr_change, velocity, 1)[0])

    scenario_rate_change_pp = 1.0
    bull_velocity_delta = beta_cbr_to_velocity * scenario_rate_change_pp
    bear_velocity_delta = beta_cbr_to_velocity * (-scenario_rate_change_pp)

    volatility_penalty = (avg_forward_vol_index - 50.0) / 100.0
    inflation_penalty = avg_forward_inflation / 100.0

    bull_case_velocity = baseline_velocity + bull_velocity_delta - volatility_penalty
    bear_case_velocity = baseline_velocity + bear_velocity_delta - (inflation_penalty * 0.5)

    bull_case_revenue_delta_kes_bn = latest_transaction_value * (bull_velocity_delta / 100.0)
    bear_case_revenue_delta_kes_bn = latest_transaction_value * (bear_velocity_delta / 100.0)

    results = {
        "scenario_analysis": {
            "base_assumptions": {
                "latest_cbr_pct": _safe_float(latest_cbr, 2),
                "baseline_mpesa_transaction_velocity_pct": _safe_float(baseline_velocity),
                "latest_transaction_value_kes_bn": _safe_float(latest_transaction_value, 2),
                "average_12m_forward_inflation_pct": _safe_float(avg_forward_inflation),
                "average_12m_forward_market_volatility_index": _safe_float(avg_forward_vol_index),
                "beta_cbr_change_to_velocity_pct_points": _safe_float(beta_cbr_to_velocity),
            },
            "bull_case_100bps_rate_hike": {
                "label": "Bull Case (+100bps rate hike)",
                "cbr_shock_bps": 100,
                "estimated_velocity_change_pct_points": _safe_float(bull_velocity_delta),
                "projected_mpesa_transaction_velocity_pct": _safe_float(bull_case_velocity),
                "estimated_transaction_revenue_impact_kes_bn": _safe_float(bull_case_revenue_delta_kes_bn, 2),
                "liquidity_signal": "Tighter collections environment" if bull_velocity_delta < 0 else "Unexpected resilience under tightening",
            },
            "bear_case_100bps_rate_cut": {
                "label": "Bear Case (-100bps rate cut)",
                "cbr_shock_bps": -100,
                "estimated_velocity_change_pct_points": _safe_float(bear_velocity_delta),
                "projected_mpesa_transaction_velocity_pct": _safe_float(bear_case_velocity),
                "estimated_transaction_revenue_impact_kes_bn": _safe_float(bear_case_revenue_delta_kes_bn, 2),
                "liquidity_signal": "Higher throughput support" if bear_velocity_delta > 0 else "Limited policy pass-through",
            },
            "advisory_note": (
                "Use the rate-hike case to pressure-test liquidity buffers, collections timing, and conservative revenue guidance. "
                "Use the rate-cut case to assess recovery upside, faster transaction conversion, and more aggressive near-term planning assumptions."
            ),
        }
    }

    output_path = artifact_dir / "scenario_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(results), indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    scenario_results = run_scenario_analysis()
    print(json.dumps(_json_ready(scenario_results), indent=2))
