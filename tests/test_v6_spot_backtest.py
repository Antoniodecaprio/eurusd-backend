import unittest
import numpy as np
import pandas as pd

from v6.spot_backtest import run_spot_anchor_backtest


class SpotAnchorBacktestTests(unittest.TestCase):
    def synthetic(self, n=1400):
        rng = np.random.default_rng(42)
        # Stationary log-price process with a persistent but mean-reverting component.
        x = np.empty(n)
        x[0] = np.log(1.08)
        theta = np.log(1.08)
        for i in range(1, n):
            x[i] = theta + 0.985 * (x[i - 1] - theta) + rng.normal(0, 0.0005)
        idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"close": np.exp(x)}, index=idx)

    def test_walk_forward_runs_and_reports_paired_models(self):
        result = run_spot_anchor_backtest(self.synthetic(), window=300, step=3, max_origins=200)
        self.assertEqual(result["status"], "completed")
        one_hour = result["results_by_horizon"]["1"]
        self.assertEqual(one_hour["status"], "evaluated")
        self.assertGreaterEqual(one_hour["baseline_previous_h1_close"]["n"], 30)
        self.assertIn("ci95", one_hour["paired_mae_difference_bootstrap"])

    def test_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            run_spot_anchor_backtest(self.synthetic(150), window=300)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run_spot_anchor_backtest(self.synthetic(), window=20)

    def test_rejects_empty_frame(self):
        with self.assertRaises(ValueError):
            run_spot_anchor_backtest(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
