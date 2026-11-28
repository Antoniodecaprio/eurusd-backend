"""Walk-forward test of a fresher spot anchor versus a one-bar-stale anchor.

This is a research benchmark, not the production OU implementation. It uses
only completed historical H1 closes and estimates an AR(1)/discrete OU process
from observations strictly before each forecast origin. BiQuote ticks are not
used because there is no verified historical tick archive in this project.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

HORIZONS = (1, 4, 24)


def _fit_ar1(log_prices: np.ndarray) -> tuple[float, float, float] | None:
    """Fit x[t+1] = intercept + phi*x[t] + epsilon on past data only."""
    x = np.asarray(log_prices, dtype=float)
    if len(x) < 30 or not np.isfinite(x).all():
        return None
    x0, x1 = x[:-1], x[1:]
    design = np.column_stack([np.ones(len(x0)), x0])
    try:
        intercept, phi = np.linalg.lstsq(design, x1, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite([intercept, phi]).all() or not 0.0 < phi < 1.0:
        return None
    residuals = x1 - (intercept + phi * x0)
    innovation_var = float(np.sum(residuals ** 2) / max(1, len(residuals) - 2))
    if not np.isfinite(innovation_var) or innovation_var <= 0:
        return None
    theta = float(intercept / (1.0 - phi))
    return float(phi), theta, innovation_var


def _forecast_log(spot_log: float, fit: tuple[float, float, float], steps: int):
    phi, theta, innovation_var = fit
    decay = phi ** steps
    mean = theta + (spot_log - theta) * decay
    variance = innovation_var * (1.0 - phi ** (2 * steps)) / (1.0 - phi ** 2)
    return mean, max(variance, 0.0)


def _metric_pack(actual, predicted, lower, upper, origin_spot):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    origin_spot = np.asarray(origin_spot, dtype=float)
    errors = predicted - actual
    actual_direction = np.sign(actual - origin_spot)
    predicted_direction = np.sign(predicted - origin_spot)
    directional = actual_direction != 0
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "directional_accuracy": (
            float(np.mean(actual_direction[directional] == predicted_direction[directional]))
            if directional.any() else None
        ),
        "interval_90_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def _bootstrap_mae_difference(baseline_errors, fresh_errors, seed=20261128, reps=2000):
    """Paired bootstrap CI for baseline absolute error minus fresh absolute error."""
    base = np.abs(np.asarray(baseline_errors, dtype=float))
    fresh = np.abs(np.asarray(fresh_errors, dtype=float))
    if len(base) < 30:
        return {"mean_baseline_minus_fresh_mae": None, "ci95": None}
    differences = base - fresh
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(reps, len(differences)))
    means = differences[draws].mean(axis=1)
    return {
        "mean_baseline_minus_fresh_mae": float(differences.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "interpretation": "Positive favours the fresher spot anchor; negative favours the stale anchor.",
        "method": "paired bootstrap over forecast origins; overlapping horizons mean this CI is descriptive, not a formal independent-sample test",
    }


def run_spot_anchor_backtest(
    candles: pd.DataFrame,
    *,
    window: int = 500,
    step: int = 4,
    max_origins: int = 1200,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """Compare identical rolling OU/AR(1) dynamics with stale vs fresh spot.

    At origin t, parameters are fitted only on closes before t. Baseline starts
    at close[t-1]; fresh-anchor variant starts at close[t]. Target is close[t+h].
    All prices are historical completed H1 closes, not executable bid/ask.
    """
    if candles is None or candles.empty or "close" not in candles.columns:
        raise ValueError("A non-empty H1 dataframe with a close column is required")
    if window < 100 or step < 1 or max_origins < 30:
        raise ValueError("Use window>=100, step>=1 and max_origins>=30")
    hs = tuple(sorted(set(int(h) for h in horizons)))
    if not hs or any(h < 1 or h > 168 for h in hs):
        raise ValueError("Horizons must be between 1 and 168 H1 bars")
    frame = candles.copy().sort_index()
    prices = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(prices) & (prices > 0)
    frame = frame.loc[valid]
    prices = prices[valid]
    if len(prices) < window + max(hs) + 31:
        raise ValueError("Insufficient valid H1 closes for the requested window and horizons")
    log_prices = np.log(prices)

    records = {h: {"actual": [], "baseline": [], "fresh": [], "lower_base": [],
                   "upper_base": [], "lower_fresh": [], "upper_fresh": [],
                   "origin": [], "baseline_origin": [], "fresh_origin": []} for h in hs}
    start = window + 1
    end = len(prices) - max(hs) - 1
    origins = list(range(start, end, step))
    if len(origins) > max_origins:
        # Retain the most recent origins; do not randomly sample the timeline.
        origins = origins[-max_origins:]

    skipped = 0
    for t in origins:
        fit = _fit_ar1(log_prices[t - window:t])
        if fit is None:
            skipped += 1
            continue
        stale_log, fresh_log = log_prices[t - 1], log_prices[t]
        for h in hs:
            actual = prices[t + h]
            mu_b, var_b = _forecast_log(stale_log, fit, h + 1)
            mu_f, var_f = _forecast_log(fresh_log, fit, h)
            z90 = 1.6448536269514722
            sd_b, sd_f = np.sqrt(var_b), np.sqrt(var_f)
            # Lognormal median forecasts; exponentiated Gaussian 90% intervals.
            records[h]["actual"].append(actual)
            records[h]["baseline"].append(float(np.exp(mu_b)))
            records[h]["fresh"].append(float(np.exp(mu_f)))
            records[h]["lower_base"].append(float(np.exp(mu_b - z90 * sd_b)))
            records[h]["upper_base"].append(float(np.exp(mu_b + z90 * sd_b)))
            records[h]["lower_fresh"].append(float(np.exp(mu_f - z90 * sd_f)))
            records[h]["upper_fresh"].append(float(np.exp(mu_f + z90 * sd_f)))
            records[h]["origin"].append(prices[t])
            records[h]["baseline_origin"].append(prices[t - 1])
            records[h]["fresh_origin"].append(prices[t])

    results = {}
    for h, r in records.items():
        if len(r["actual"]) < 30:
            results[str(h)] = {"status": "insufficient_samples", "n": len(r["actual"])}
            continue
        actual = np.asarray(r["actual"])
        base_pred, fresh_pred = np.asarray(r["baseline"]), np.asarray(r["fresh"])
        # Score direction from each strategy's own information set.
        # The baseline only knows close[t-1]; the refreshed variant knows close[t].
        baseline_metrics = _metric_pack(
            actual, base_pred, r["lower_base"], r["upper_base"], r["baseline_origin"]
        )
        fresh_metrics = _metric_pack(
            actual, fresh_pred, r["lower_fresh"], r["upper_fresh"], r["fresh_origin"]
        )
        base_mae, fresh_mae = baseline_metrics["mae"], fresh_metrics["mae"]
        results[str(h)] = {
            "status": "evaluated",
            "horizon_h1_bars": h,
            "horizon_hours_approx": h,
            "baseline_previous_h1_close": baseline_metrics,
            "fresh_current_h1_close": fresh_metrics,
            "mae_improvement_pct": (
                float((base_mae - fresh_mae) / base_mae * 100) if base_mae else None
            ),
            "paired_mae_difference_bootstrap": _bootstrap_mae_difference(
                base_pred - actual, fresh_pred - actual
            ),
        }

    return {
        "status": "completed" if any(x.get("status") == "evaluated" for x in results.values()) else "insufficient_data",
        "method": "walk_forward_discrete_OU_AR1_same_parameters_different_spot_anchor",
        "symbol": "EUR/USD",
        "bar_interval": "H1",
        "calibration_window_bars": window,
        "origin_step_bars": step,
        "forecast_origins_evaluated": int(sum(
            x.get("baseline_previous_h1_close", {}).get("n", 0) for x in results.values()
            if x.get("status") == "evaluated"
        ) / max(1, len([x for x in results.values() if x.get("status") == "evaluated"]))),
        "origins_skipped_non_mean_reverting_fit": skipped,
        "first_candle_utc": frame.index[0].isoformat(),
        "last_candle_utc": frame.index[-1].isoformat(),
        "results_by_horizon": results,
        "limitations": [
            "This is a transparent benchmark implementation, not a reconstruction of the unavailable production OU engine.",
            "Fresh spot is proxied by the latest completed H1 close; no historical BiQuote tick archive is available here.",
            "Baseline is the immediately preceding H1 close, so the experiment measures a one-bar spot refresh only.",
            "Historical close prices are not executable bid/ask prices; spread, slippage, swaps and commissions are excluded.",
            "Forecast origins overlap; bootstrap intervals are descriptive and should not be treated as formal independent-sample significance tests.",
            "A better midpoint forecast does not establish profitable trading performance.",
        ],
        "trading_enabled": False,
        "historical_data_modified": False,
    }
