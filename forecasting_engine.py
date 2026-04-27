from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.statespace.sarimax import SARIMAX

from diagnostics import build_model_validation_summary, save_diagnostics
from scenario_simulator import run_scenario_analysis

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


class KenyaEconomicIntelligenceEngine:
    """FP&A macro-fintech forecasting engine for Kenya.

    The current implementation uses synthetic monthly data shaped like CBK / KNBS
    indicators so the repository can run end-to-end without external credentials.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _month_end_range(start: str = "2021-01-31", periods: int = 60) -> pd.DatetimeIndex:
        return pd.date_range(start=start, periods=periods, freq="M")

    @staticmethod
    def _cbr_schedule(date: pd.Timestamp) -> float:
        if date < pd.Timestamp("2021-07-31"):
            return 7.00
        if date < pd.Timestamp("2022-06-30"):
            return 7.50
        if date < pd.Timestamp("2022-12-31"):
            return 8.25
        if date < pd.Timestamp("2023-07-31"):
            return 9.50
        if date < pd.Timestamp("2024-01-31"):
            return 12.50
        if date < pd.Timestamp("2025-01-31"):
            return 13.00
        if date < pd.Timestamp("2025-07-31"):
            return 12.75
        return 11.25

    def generate_mock_cbk_data(self, start: str = "2021-01-31", periods: int = 60) -> pd.DataFrame:
        dates = self._month_end_range(start=start, periods=periods)
        cbr = np.array([self._cbr_schedule(d) for d in dates], dtype=float)

        seasonal_pattern = np.array([28, 16, 12, 8, 5, 0, 6, 10, 18, 22, 35, 58], dtype=float)
        fintech_campaign_bumps = np.array([6 if d.month in (3, 9, 12) else 0 for d in dates], dtype=float)

        mobile_money = []
        current_value = 455.0
        prior_cbr = cbr[0]

        for i, d in enumerate(dates):
            base_trend = 4.8 + 0.08 * i
            seasonal = seasonal_pattern[d.month - 1]
            policy_drag = -3.4 * max(cbr[i] - 8.0, 0)
            rate_shock_drag = (-28.0 * max(cbr[i] - prior_cbr, 0)) + (10.0 * max(prior_cbr - cbr[i], 0))
            macro_shock = self.rng.normal(0, 11)
            digitisation_tailwind = 5.0 * np.sin((2 * np.pi * i / 12) + 0.9)

            current_value = max(
                380,
                current_value
                + base_trend
                + seasonal
                + fintech_campaign_bumps[i]
                + digitisation_tailwind
                + policy_drag
                + rate_shock_drag
                + macro_shock,
            )
            mobile_money.append(current_value)
            prior_cbr = cbr[i]

        mobile_money = np.array(mobile_money)
        mobile_money_growth = pd.Series(mobile_money).pct_change().fillna(0) * 100

        cpi = [100.0]
        for i in range(1, len(dates)):
            seasonal_infl = 0.07 * np.sin(2 * np.pi * i / 12)
            pass_through = 0.055 * mobile_money_growth.iloc[i - 1]
            rate_cooling = -0.035 * max(cbr[i - 1] - 8.5, 0)
            random_infl_shock = self.rng.normal(0, 0.10)
            monthly_inflation = 0.42 + seasonal_infl + pass_through + rate_cooling + random_infl_shock
            monthly_inflation = float(np.clip(monthly_inflation, 0.08, 1.15))
            cpi.append(cpi[-1] * (1 + monthly_inflation / 100))

        df = pd.DataFrame(
            {
                "date": dates,
                "cpi": np.round(cpi, 2),
                "mobile_money_value_kes_bn": np.round(mobile_money, 2),
                "cbr": np.round(cbr, 2),
            }
        )

        df.loc[10, "mobile_money_value_kes_bn"] = np.nan
        df.loc[24, "cpi"] = np.nan
        df.loc[37, "cbr"] = np.nan
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"])
        cleaned = cleaned.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        numeric_cols = ["cpi", "mobile_money_value_kes_bn", "cbr"]
        for col in numeric_cols:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(method="linear", limit_direction="both")
        cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()

        cleaned["inflation_mom_pct"] = cleaned["cpi"].pct_change() * 100
        cleaned["inflation_yoy_pct"] = cleaned["cpi"].pct_change(12) * 100
        cleaned["mpesa_growth_pct"] = cleaned["mobile_money_value_kes_bn"].pct_change() * 100
        cleaned["cbr_change_pp"] = cleaned["cbr"].diff()

        for col in ["inflation_mom_pct", "inflation_yoy_pct", "mpesa_growth_pct"]:
            lower = cleaned[col].quantile(0.02)
            upper = cleaned[col].quantile(0.98)
            cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

        return cleaned

    def fit_sarima(self, df: pd.DataFrame, horizon: int = 12) -> Dict[str, Any]:
        target = df.set_index("date")["inflation_yoy_pct"].dropna()
        train = target.iloc[:-horizon]
        test = target.iloc[-horizon:]

        candidate_specs = [
            ((1, 0, 1), (1, 0, 1, 12)),
            ((1, 1, 1), (1, 0, 0, 12)),
            ((2, 0, 1), (0, 1, 1, 12)),
        ]

        fitted = None
        chosen_spec = None
        for order, seasonal_order in candidate_specs:
            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                chosen_spec = {"order": order, "seasonal_order": seasonal_order}
                break
            except Exception:
                continue

        if fitted is None or chosen_spec is None:
            raise RuntimeError("SARIMA fitting failed across all candidate specifications.")

        holdout_forecast = fitted.get_forecast(steps=horizon)
        holdout_pred = pd.Series(holdout_forecast.predicted_mean.values, index=test.index, name="forecast")
        holdout_ci = holdout_forecast.conf_int()

        mae = float(np.mean(np.abs(test.values - holdout_pred.values)))
        rmse = float(np.sqrt(np.mean((test.values - holdout_pred.values) ** 2)))

        full_model = SARIMAX(
            target,
            order=chosen_spec["order"],
            seasonal_order=chosen_spec["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        full_fitted = full_model.fit(disp=False)
        future_forecast = full_fitted.get_forecast(steps=horizon)
        future_ci = future_forecast.conf_int()
        future_dates = pd.date_range(target.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")

        holdout_df = pd.DataFrame(
            {
                "date": test.index,
                "actual_inflation_yoy_pct": test.values,
                "forecast_inflation_yoy_pct": holdout_pred.values,
                "lower_ci": holdout_ci.iloc[:, 0].values,
                "upper_ci": holdout_ci.iloc[:, 1].values,
            }
        )

        future_df = pd.DataFrame(
            {
                "date": future_dates,
                "forecast_inflation_yoy_pct": future_forecast.predicted_mean.values,
                "lower_ci": future_ci.iloc[:, 0].values,
                "upper_ci": future_ci.iloc[:, 1].values,
            }
        )

        return {
            "model_type": "SARIMA",
            "chosen_specification": chosen_spec,
            "holdout_mae": round(mae, 4),
            "holdout_rmse": round(rmse, 4),
            "historical_actuals": target.reset_index().rename(columns={"inflation_yoy_pct": "actual_inflation_yoy_pct"}),
            "holdout_forecast": holdout_df,
            "future_forecast": future_df,
        }

    def fit_garch(self, df: pd.DataFrame, horizon: int = 12) -> Dict[str, Any]:
        series = df.set_index("date")["mpesa_growth_pct"].dropna()
        model = arch_model(series, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        fitted = model.fit(disp="off")

        conditional_vol = pd.Series(fitted.conditional_volatility, index=series.index, name="conditional_volatility")
        standardized_index = 50 + 15 * ((conditional_vol - conditional_vol.mean()) / conditional_vol.std(ddof=0))
        standardized_index = standardized_index.clip(lower=0, upper=100).rename("market_volatility_index")

        forecast = fitted.forecast(horizon=horizon, reindex=False)
        future_vol = np.sqrt(forecast.variance.values[-1])
        future_dates = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
        future_index = 50 + 15 * ((future_vol - conditional_vol.mean()) / conditional_vol.std(ddof=0))
        future_index = np.clip(future_index, 0, 100)

        hist_df = pd.DataFrame(
            {
                "date": conditional_vol.index,
                "conditional_volatility": conditional_vol.values,
                "market_volatility_index": standardized_index.values,
            }
        )

        future_df = pd.DataFrame(
            {
                "date": future_dates,
                "forecast_conditional_volatility": future_vol,
                "forecast_market_volatility_index": future_index,
            }
        )

        current_index = float(hist_df["market_volatility_index"].iloc[-1])
        regime = "High" if current_index >= 67 else "Moderate" if current_index >= 45 else "Low"

        return {
            "model_type": "GARCH(1,1)",
            "current_market_volatility_index": round(current_index, 2),
            "volatility_regime": regime,
            "historical_volatility": hist_df,
            "future_volatility": future_df,
        }

    def advisory_note_for_cfos(self, df: pd.DataFrame) -> Dict[str, Any]:
        sample = df[["date", "mpesa_growth_pct", "cbr_change_pp", "cbr", "inflation_yoy_pct", "mobile_money_value_kes_bn"]].dropna().copy()
        sample["lagged_inflation"] = sample["inflation_yoy_pct"].shift(1)
        sample = sample.dropna().copy()

        y = sample["mpesa_growth_pct"]
        x = add_constant(sample[["cbr_change_pp", "cbr", "lagged_inflation"]])
        model = OLS(y, x).fit()

        cbr_change_beta = float(model.params["cbr_change_pp"])
        latest_value = float(sample["mobile_money_value_kes_bn"].iloc[-1])
        latest_growth = float(sample["mpesa_growth_pct"].iloc[-1])

        hike_100bps_velocity_impact = cbr_change_beta
        cut_100bps_velocity_impact = -cbr_change_beta
        monthly_uplift_kes_bn = latest_value * (cut_100bps_velocity_impact / 100)
        annualised_uplift_kes_bn = monthly_uplift_kes_bn * 12

        return {
            "model_type": "OLS policy pass-through proxy",
            "r_squared": round(float(model.rsquared), 4),
            "latest_transaction_growth_pct": round(latest_growth, 3),
            "beta_cbr_change_to_velocity_pct_points": round(cbr_change_beta, 4),
            "impact_of_100bps_hike_on_velocity_pct_points": round(hike_100bps_velocity_impact, 4),
            "impact_of_100bps_cut_on_velocity_pct_points": round(cut_100bps_velocity_impact, 4),
            "estimated_monthly_transaction_uplift_kes_bn_after_100bps_cut": round(monthly_uplift_kes_bn, 2),
            "estimated_annual_transaction_uplift_kes_bn_after_100bps_cut": round(annualised_uplift_kes_bn, 2),
            "advisory_message": (
                "Mobile money velocity should be tracked as a leading indicator for liquidity crunches, revenue fragility, and human capital risk."
            ),
        }

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            df = value.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
            return df.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, dict):
            return {k: KenyaEconomicIntelligenceEngine._json_ready(v) for k, v in value.items()}
        if isinstance(value, list):
            return [KenyaEconomicIntelligenceEngine._json_ready(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    def strategic_note(self, validation: Dict[str, Any], sarima: Dict[str, Any], garch: Dict[str, Any], advisory: Dict[str, Any]) -> Dict[str, str]:
        inflation_stationary = validation["financial_model_validation"]["stationarity"]["inflation_yoy"]["stationary"]
        granger_detected = validation["financial_model_validation"]["granger_causality"]["granger_causality_detected"]
        future_avg_inflation = round(float(sarima["future_forecast"]["forecast_inflation_yoy_pct"].mean()), 2)

        return {
            "policy_readout": (
                f"Inflation YoY is {'stationary' if inflation_stationary else 'non-stationary'}, which tells finance leadership whether the forecasting base is stable or drifting. "
                f"Current market volatility is {garch['volatility_regime'].lower()}, so planning should emphasize either resilience or shock containment."
            ),
            "commercial_implication": (
                "Mobile money growth shows predictive content for inflation shifts. "
                if granger_detected
                else "Mobile money growth does not clear a strict causality threshold in this run, so treat it as directional rather than deterministic. "
            )
            +
