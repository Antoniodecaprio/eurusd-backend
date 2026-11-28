import unittest
from unittest.mock import patch, Mock

import pandas as pd
import requests

from v6.biquote import BiquoteError, fetch_candles


class BiquoteAdapterTests(unittest.TestCase):
    def payload(self):
        return {"symbol":"EURUSD","interval":"15m","bars":[
            {"openTime":"2026-11-27T09:15:00Z","open":1.1,"high":1.101,"low":1.099,"close":1.1005,"isOpen":False},
            {"openTime":"2026-11-27T09:30:00Z","open":1.1005,"high":1.102,"low":1.1,"close":1.101,"isOpen":True},
        ]}

    @patch("v6.biquote.requests.get")
    def test_completed_candles_only_and_utc_index(self, get):
        response=Mock()
        response.json.return_value=self.payload()
        get.return_value=response
        frame=fetch_candles("M15")
        self.assertEqual(len(frame),1)
        self.assertEqual(frame.index[0],pd.Timestamp("2026-11-27T09:15:00Z"))
        self.assertEqual(frame.iloc[0]["Close"],1.1005)
        self.assertEqual(frame.attrs["price_basis"],"mid")
        self.assertIn("Yahoo bootstrap",frame.attrs["upstream_lineage"])
        self.assertEqual(get.call_args.kwargs["params"]["interval"],"15m")

    @patch("v6.biquote.requests.get")
    def test_h1_uses_168_bar_window(self,get):
        response=Mock()
        response.json.return_value={"bars":[{"openTime":"2026-11-27T09:00:00Z","open":1.1,"high":1.1,"low":1.1,"close":1.1,"isOpen":False}]}
        get.return_value=response
        fetch_candles("H1",limit=500)
        self.assertEqual(get.call_args.kwargs["params"]["limit"],168)
        self.assertEqual(get.call_args.kwargs["params"]["interval"],"1h")

    def test_rejects_unsupported_timeframe(self):
        with self.assertRaises(ValueError):
            fetch_candles("D1")

    @patch("v6.biquote.requests.get")
    def test_missing_bars_is_error(self,get):
        response=Mock()
        response.json.return_value={"symbol":"EURUSD"}
        get.return_value=response
        with self.assertRaises(BiquoteError):
            fetch_candles("M15")

    @patch("v6.biquote.requests.get")
    def test_timeout_is_wrapped(self,get):
        get.side_effect=requests.Timeout()
        with self.assertRaises(BiquoteError):
            fetch_candles("M15")


if __name__ == "__main__":
    unittest.main()
