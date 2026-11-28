import unittest
from datetime import datetime, timezone, timedelta

from v6.forecast_context import build_forecast_market_context


class ForecastMarketContextTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

    def quote(self, **changes):
        value = {
            "provider": "biquote", "bid": 1.0854, "ask": 1.0855,
            "timestamp_utc": self.now.isoformat(), "market_open": True,
            "provider_stale": False, "source_verified": False,
        }
        value.update(changes)
        return value

    def test_fresh_quote_becomes_candidate_only(self):
        result = build_forecast_market_context(self.quote(), now=self.now)
        self.assertEqual(result["status"], "ready")
        self.assertAlmostEqual(result["forecast_handoff"]["spot_candidate"], 1.08545)
        self.assertFalse(result["forecast_handoff"]["automatic_model_injection"])
        self.assertFalse(result["historical_data_modified"])
        self.assertFalse(result["ou_parameters_modified"])

    def test_missing_quote_fails_closed(self):
        result = build_forecast_market_context(None, now=self.now)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["forecast_handoff"]["spot_candidate"])

    def test_old_quote_cannot_be_handed_to_model(self):
        old = (self.now - timedelta(minutes=2)).isoformat()
        result = build_forecast_market_context(self.quote(timestamp_utc=old), now=self.now)
        self.assertEqual(result["status"], "not_ready")
        self.assertIsNone(result["forecast_handoff"]["spot_candidate"])

    def test_closed_market_cannot_be_handed_to_model(self):
        result = build_forecast_market_context(self.quote(market_open=False), now=self.now)
        self.assertEqual(result["status"], "not_ready")

    def test_provider_stale_flag_cannot_be_overridden(self):
        result = build_forecast_market_context(self.quote(provider_stale=True), now=self.now)
        self.assertEqual(result["status"], "not_ready")

    def test_invalid_spread_fails_closed(self):
        result = build_forecast_market_context(self.quote(ask=1.08), now=self.now)
        self.assertEqual(result["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
