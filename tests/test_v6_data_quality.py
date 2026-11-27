import unittest

import pandas as pd

from v6.data_quality import normalise_candles, resample_completed


class CandleQualityTests(unittest.TestCase):
    def make_frame(self, starts, prices=None):
        prices = prices or [1.10] * len(starts)
        return pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 0.001 for p in prices],
                "Low": [p - 0.001 for p in prices],
                "Close": prices,
            },
            index=pd.DatetimeIndex(starts),
        )

    def test_incomplete_m15_candle_is_removed(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=8, freq="15min")
        frame = self.make_frame(starts)
        cleaned, quality = normalise_candles(
            frame, "M15", "unit-test",
            now=pd.Timestamp("2026-11-27T09:52:00Z"),
            minimum_candles=1,
        )
        self.assertEqual(cleaned.index[-1], pd.Timestamp("2026-11-27T09:30:00Z"))
        self.assertFalse(quality.stale)

    def test_duplicate_and_invalid_ohlc_rows_are_removed(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        frame = self.make_frame(starts)
        frame = pd.concat([frame, frame.iloc[[-1]]])
        frame.loc[starts[1], "High"] = 1.0  # Below open/close: invalid envelope.
        cleaned, quality = normalise_candles(
            frame, "M15", "unit-test",
            now=pd.Timestamp("2026-11-27T09:00:00Z"),
            minimum_candles=1,
        )
        self.assertEqual(quality.duplicate_count, 1)
        self.assertEqual(quality.invalid_count, 1)
        self.assertEqual(len(cleaned), 3)

    def test_resampling_rejects_partial_h1_bars(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=7, freq="15min")
        frame = self.make_frame(starts)
        hourly = resample_completed(
            frame, "M15", "H1", now=pd.Timestamp("2026-11-27T09:52:00Z")
        )
        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly.index[0], pd.Timestamp("2026-11-27T08:00:00Z"))


    def test_resampling_accepts_provider_title_case_columns(self):
        starts = pd.date_range("2026-11-27T08:00:00Z", periods=4, freq="15min")
        frame = self.make_frame(starts)
        hourly = resample_completed(
            frame, "M15", "H1", now=pd.Timestamp("2026-11-27T09:05:00Z")
        )
        self.assertEqual(len(hourly), 1)
        self.assertEqual(float(hourly.iloc[0]["close"]), 1.10)

    def test_stale_data_is_not_ready(self):
        starts = pd.date_range("2026-11-27T01:00:00Z", periods=4, freq="15min")
        frame = self.make_frame(starts)
        _, quality = normalise_candles(
            frame, "M15", "unit-test",
            now=pd.Timestamp("2026-11-27T04:00:00Z"),
            minimum_candles=1,
        )
        self.assertTrue(quality.stale)
        self.assertFalse(quality.ready)


if __name__ == "__main__":
    unittest.main()
