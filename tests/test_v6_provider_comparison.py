import unittest
from unittest.mock import patch

import pandas as pd

from v6.provider_comparison import compare_provider_frames, fetch_provider_evidence


class ProviderComparisonTests(unittest.TestCase):
    def frame(self, starts, closes=None):
        closes = closes or [1.10] * len(starts)
        return pd.DataFrame(
            {
                "Open": closes,
                "High": [value + 0.0002 for value in closes],
                "Low": [value - 0.0002 for value in closes],
                "Close": closes,
            },
            index=pd.DatetimeIndex(starts),
        )

    def test_both_providers_are_preserved_and_compared(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        result = compare_provider_frames(
            self.frame(starts, [1.10] * 4),
            self.frame(starts, [1.10006] * 4),
            "M15",
            now=pd.Timestamp("2026-11-27T09:05:00Z"),
        )
        self.assertEqual(set(result["providers"]), {"Twelve Data", "Yahoo Finance"})
        self.assertEqual(result["comparison"]["matched_candles"], 4)
        self.assertAlmostEqual(result["comparison"]["latest_close_difference_pips"], 0.6)
        self.assertEqual(result["comparison"]["status"], "compared")

    @patch("v6.provider_comparison.yf.Ticker")
    @patch("v6.provider_comparison.fetch_twelve_data_candles")
    def test_twelve_data_failure_does_not_hide_yahoo(self, twelve, ticker):
        twelve.side_effect = RuntimeError("Twelve Data unavailable")
        ticker.return_value.history.return_value = self.frame(
            pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        )
        result = fetch_provider_evidence("M15", now=pd.Timestamp("2026-11-27T09:05:00Z"))
        self.assertEqual(result["providers"]["Yahoo Finance"]["status"], "available")
        self.assertEqual(result["providers"]["Twelve Data"]["status"], "unavailable")
        self.assertEqual(result["comparison"]["status"], "insufficient_provider_data")

    @patch("v6.provider_comparison.yf.Ticker")
    @patch("v6.provider_comparison.fetch_twelve_data_candles")
    def test_yahoo_failure_does_not_hide_twelve_data(self, twelve, ticker):
        twelve.return_value = self.frame(
            pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        )
        ticker.return_value.history.side_effect = RuntimeError("Yahoo unavailable")
        result = fetch_provider_evidence("M15", now=pd.Timestamp("2026-11-27T09:05:00Z"))
        self.assertEqual(result["providers"]["Twelve Data"]["status"], "available")
        self.assertEqual(result["providers"]["Yahoo Finance"]["status"], "unavailable")

    @patch("v6.provider_comparison.yf.Ticker")
    @patch("v6.provider_comparison.fetch_twelve_data_candles")
    def test_both_provider_failures_are_reported_without_prices(self, twelve, ticker):
        twelve.side_effect = RuntimeError("Twelve Data unavailable")
        ticker.return_value.history.side_effect = RuntimeError("Yahoo unavailable")
        result = fetch_provider_evidence("M15", now=pd.Timestamp("2026-11-27T09:05:00Z"))
        self.assertEqual(result["comparison"]["status"], "unavailable")
        self.assertIsNone(result["comparison"]["latest_close_difference_pips"])
        self.assertEqual(result["providers"]["Twelve Data"]["observations"], [])
        self.assertEqual(result["providers"]["Yahoo Finance"]["observations"], [])

    def test_different_candle_timestamps_are_not_matched(self):
        a = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        b = pd.date_range("2026-11-27T08:15:00Z", periods=4, freq="15min")
        result = compare_provider_frames(self.frame(a), self.frame(b), "M15",
                                         now=pd.Timestamp("2026-11-27T09:30:00Z"))
        self.assertEqual(result["comparison"]["matched_candles"], 3)
        self.assertNotEqual(
            result["providers"]["Twelve Data"]["observations"][-1]["candle_open_utc"],
            result["providers"]["Yahoo Finance"]["observations"][-1]["candle_open_utc"],
        )

    def test_incomplete_candles_are_excluded(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=5, freq="15min")
        result = compare_provider_frames(self.frame(starts), self.frame(starts), "M15",
                                         now=pd.Timestamp("2026-11-27T09:00:00Z"))
        self.assertEqual(len(result["providers"]["Twelve Data"]["observations"]), 4)

    def test_invalid_ohlc_rows_are_rejected(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        invalid = self.frame(starts)
        invalid.loc[starts[1], "High"] = 1.0
        result = compare_provider_frames(invalid, self.frame(starts), "M15",
                                         now=pd.Timestamp("2026-11-27T09:05:00Z"))
        self.assertEqual(result["providers"]["Twelve Data"]["quality"]["invalid_count"], 1)
        self.assertEqual(result["comparison"]["matched_candles"], 3)

    def test_one_provider_never_produces_consensus(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        result = compare_provider_frames(self.frame(starts), None, "M15",
                                         now=pd.Timestamp("2026-11-27T09:05:00Z"))
        self.assertEqual(result["comparison"]["status"], "insufficient_provider_data")
        self.assertIsNone(result["comparison"]["consensus_close"])
        
    def test_disagreement_is_reported_not_resolved_by_averaging(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        result = compare_provider_frames(self.frame(starts), self.frame(starts, [1.101] * 4),
                                         "M15", now=pd.Timestamp("2026-11-27T09:05:00Z"))
        self.assertGreater(result["comparison"]["latest_close_difference_pips"], 1.0)
        self.assertIsNone(result["comparison"]["consensus_close"])


    def test_m15_deviation_within_tolerance_is_classified_without_hiding_value(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        result = compare_provider_frames(
            self.frame(starts), self.frame(starts, [1.10005] * 4), "M15",
            now=pd.Timestamp("2026-11-27T09:05:00Z"),
        )
        comparison = result["comparison"]
        self.assertEqual(comparison["deviation_status"], "within_tolerance")
        self.assertAlmostEqual(comparison["latest_close_difference_pips"], 0.5)
        self.assertEqual(comparison["deviation_tolerance_pips"], 0.5)

    def test_m15_deviation_above_tolerance_is_flagged_not_rejected(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        result = compare_provider_frames(
            self.frame(starts), self.frame(starts, [1.10006] * 4), "M15",
            now=pd.Timestamp("2026-11-27T09:05:00Z"),
        )
        comparison = result["comparison"]
        self.assertEqual(comparison["deviation_status"], "above_tolerance")
        self.assertAlmostEqual(comparison["latest_close_difference_pips"], 0.6)
        self.assertEqual(comparison["matched_candles"], 4)
        self.assertIsNone(comparison["consensus_close"])

    def test_h1_uses_its_own_deviation_tolerance(self):
        starts = pd.date_range("2026-11-27T04:00:00Z", periods=4, freq="1h")
        result = compare_provider_frames(
            self.frame(starts), self.frame(starts, [1.10009] * 4), "H1",
            now=pd.Timestamp("2026-11-27T09:00:00Z"),
        )
        comparison = result["comparison"]
        self.assertEqual(comparison["deviation_status"], "within_tolerance")
        self.assertEqual(comparison["deviation_tolerance_pips"], 1.0)

    def test_no_deviation_classification_without_matching_candles(self):
        a = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        b = pd.date_range("2026-11-27T10:00:00Z", periods=4, freq="15min")
        result = compare_provider_frames(
            self.frame(a), self.frame(b), "M15",
            now=pd.Timestamp("2026-11-27T11:00:00Z"),
        )
        self.assertEqual(result["comparison"]["deviation_status"], "not_comparable")
        self.assertIsNone(result["comparison"]["deviation_tolerance_pips"])

if __name__ == "__main__":
    unittest.main()
