import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from v6_app import app


class ProviderEvidenceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("v6_app.fetch_provider_evidence")
    def test_endpoint_returns_provider_comparison(self, fetch):
        fetch.return_value = {
            "symbol": "EUR/USD",
            "timeframe": "M15",
            "providers": {},
            "comparison": {
                "status": "compared",
                "deviation_status": "above_tolerance",
                "latest_close_difference_pips": 0.8,
            },
        }
        response = self.client.get("/v6/provider-evidence?timeframe=m15")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["comparison"]["deviation_status"], "above_tolerance")
        fetch.assert_called_once_with("M15")

    def test_endpoint_rejects_unsupported_timeframe(self):
        response = self.client.get("/v6/provider-evidence?timeframe=M5")
        self.assertEqual(response.status_code, 400)

    @patch("v6_app.fetch_provider_evidence")
    def test_endpoint_returns_502_on_unexpected_provider_failure(self, fetch):
        fetch.side_effect = RuntimeError("provider failure")
        response = self.client.get("/v6/provider-evidence")
        self.assertEqual(response.status_code, 502)
        self.assertIn("RuntimeError", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
