import unittest
from fastapi.testclient import TestClient
from v6_app import app

class V6ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_status_disables_execution_and_forecasts_until_ready(self):
        data = self.client.get("/v6/status").json()
        self.assertFalse(data["mt5_execution_enabled"])
        self.assertFalse(data["live_trading_enabled"])
        self.assertFalse(data["forecast_models_enabled"])
        self.assertEqual(data["demo_risk_limit_pct"], 0.5)


    def test_market_quality_returns_compact_summary_without_candles(self):
        from unittest.mock import patch

        fake = {
            "symbol": "EURUSD=X",
            "source": "test",
            "retrieved_at_utc": "2026-11-27T00:00:00+00:00",
            "timeframes": {
                "M15": {
                    "status": "ready",
                    "metadata": {
                        "limitations": ["research feed"],
                        "quality": {
                            "candle_count": 200,
                            "last_candle_close_utc": "2026-11-27T00:00:00+00:00",
                            "age_seconds": 30,
                            "stale": False,
                            "missing_interval_count": 4,
                            "missing_interval_interpretation": "Raw elapsed-time gaps; not session-aware.",
                            "duplicate_count": 0,
                            "invalid_count": 0,
                            "ready": True,
                            "reasons": [],
                        },
                    },
                    "candles": [{"time": "should-not-be-returned"}],
                },
            },
        }
        with patch("v6_app.fetch_market_snapshot", return_value=fake):
            response = self.client.get("/v6/market-quality")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["timeframes"]["M15"]["candle_count"], 200)
        self.assertNotIn("candles", data["timeframes"]["M15"])
        self.assertFalse(data["trading_enabled"])

    def test_root_identifies_staging(self):
        data = self.client.get("/").json()
        self.assertEqual(data["version"], "6.0.0-dev")
        self.assertEqual(data["environment"], "staging")

if __name__ == "__main__":
    unittest.main()
