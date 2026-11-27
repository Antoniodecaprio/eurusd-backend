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

    def test_root_identifies_staging(self):
        data = self.client.get("/").json()
        self.assertEqual(data["version"], "6.0.0-dev")
        self.assertEqual(data["environment"], "staging")

if __name__ == "__main__":
    unittest.main()
