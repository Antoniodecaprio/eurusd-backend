import unittest
from unittest.mock import patch

import pandas as pd

from v6.market_data import MarketDataError, fetch_timeframe


class MarketDataAdapterTests(unittest.TestCase):
    def make_provider_frame(self, periods=160, freq="15min"):
        index = pd.date_range(
            "2026-11-25T00:00:00Z", periods=periods, freq=freq
        )
        close = [1.10 + i * 0.00001 for i in range(periods)]
        return pd.DataFrame(
            {
                "Open": close,
                "High": [p + 0.0002 for p in close],
                "Low": [p - 0.0002 for p in close],
                "Close": close,
                "Volume": [0] * periods,
            },
            index=index,
        )

    @patch("v6.market_data._download")
    def test_m15_adapter_returns_completed_candles_and_source_metadata(self, download):
        download.return_value = self.make_provider_frame()
        now = pd.Timestamp("2026-11-26T16:05:00Z")
        candles, metadata = fetch_timeframe(
            "M15", now=now, minimum_candles=100
        )
        self.assertGreaterEqual(len(candles), 100)
        self.assertLessEqual(candles.index[-1] + pd.Timedelta(minutes=15), now)
        self.assertEqual(metadata["source"], "Yahoo Finance (yfinance)")
        self.assertEqual(metadata["symbol"], "EURUSD=X")
        self.assertTrue(metadata["quality"]["ready"])

    @patch("v6.market_data._download")
    def test_empty_provider_response_raises_market_data_error(self, download):
        download.side_effect = MarketDataError("provider unavailable")
        with self.assertRaises(MarketDataError):
            fetch_timeframe("M15", minimum_candles=10)

    def test_unsupported_timeframe_is_rejected(self):
        with self.assertRaises(ValueError):
            fetch_timeframe("M5")


if __name__ == "__main__":
    unittest.main()
