import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

import requests

from v6.biquote_live import (
    BiquoteLiveError, BiquoteSignalRStream, LiveQuoteStore, fetch_live_quote, validate_quote,
)


class BiquoteLiveTests(unittest.TestCase):
    def payload(self, **updates):
        value = {
            "symbol": "EURUSD", "bid": 1.08542, "ask": 1.08548,
            "timestamp": datetime.now(timezone.utc).isoformat(), "marketState": "open", "stale": False,
        }
        value.update(updates)
        return value

    def test_valid_quote_has_spread_and_provenance(self):
        quote = validate_quote(self.payload())
        self.assertAlmostEqual(quote["spread_pips"], 0.6, places=6)
        self.assertFalse(quote["source_verified"])
        self.assertFalse(quote["historical_consensus_eligible"])
        self.assertEqual(quote["data_type"], "live_quote")

    def test_rejects_ask_below_bid(self):
        with self.assertRaisesRegex(BiquoteLiveError, "ask_below_bid"):
            validate_quote(self.payload(bid=1.2, ask=1.1))

    def test_rejects_non_positive_prices(self):
        with self.assertRaises(BiquoteLiveError):
            validate_quote(self.payload(bid=0))

    def test_rejects_missing_timestamp(self):
        payload = self.payload()
        payload.pop("timestamp")
        with self.assertRaisesRegex(BiquoteLiveError, "quote_timestamp_missing"):
            validate_quote(payload)

    def test_marks_old_quote_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        quote = validate_quote(self.payload(timestamp=old))
        self.assertTrue(quote["is_stale"])
        self.assertFalse(quote["is_live"])

    def test_rejects_future_timestamp(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(BiquoteLiveError, "quote_timestamp_in_future"):
            validate_quote(self.payload(timestamp=future))

    def test_provider_closed_market_is_not_reported_live(self):
        quote = validate_quote(self.payload(marketState="closed"))
        self.assertFalse(quote["market_open"])
        self.assertFalse(quote["is_live"])

    def test_provider_stale_flag_is_respected(self):
        quote = validate_quote(self.payload(stale=True))
        self.assertTrue(quote["provider_stale"])
        self.assertTrue(quote["is_stale"])
        self.assertFalse(quote["is_live"])

    def test_rejects_wrong_symbol(self):
        with self.assertRaisesRegex(BiquoteLiveError, "unexpected_symbol"):
            validate_quote(self.payload(symbol="GBPUSD"))

    @patch("v6.biquote_live.requests.get")
    def test_rest_quote_fetch(self, get):
        response = Mock()
        response.json.return_value = self.payload()
        get.return_value = response
        quote = fetch_live_quote()
        self.assertEqual(quote["provider"], "biquote")
        self.assertEqual(get.call_args.args[0], "https://biquote.io/api/EURUSD")

    @patch("v6.biquote_live.requests.get")
    def test_rest_timeout_is_safe(self, get):
        get.side_effect = requests.Timeout()
        with self.assertRaisesRegex(BiquoteLiveError, "biquote_quote_timeout"):
            fetch_live_quote()

    def test_signalr_stream_validates_tick_and_updates_cache(self):
        store = LiveQuoteStore()
        stream = BiquoteSignalRStream(store.update)
        stream.last_message_at = datetime.now(timezone.utc).isoformat()
        tick = validate_quote(self.payload())
        stream.on_quote(tick)
        self.assertEqual(store.get()["provider"], "biquote")

    def test_store_returns_copy(self):
        store = LiveQuoteStore()
        quote = validate_quote(self.payload())
        store.update(quote)
        result = store.get()
        result["bid"] = 0
        self.assertNotEqual(store.get()["bid"], 0)


if __name__ == "__main__":
    unittest.main()
