import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import main_v5_3


class EconomicCalendarRiskTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.days = 7

    def calendar(self, events, **overrides):
        result = {
            "status": "available",
            "provider": "ShowMyTrades",
            "server_time_utc": self.now.isoformat(),
            "coverage_start_utc": (self.now - timedelta(hours=6)).isoformat(),
            "coverage_end_utc": (self.now + timedelta(days=self.days)).isoformat(),
            "events": events,
            "event_count": len(events),
        }
        result.update(overrides)
        return result

    def event(self, currency="USD", impact="high"):
        return {
            "currency": currency,
            "impact": impact,
            "datetime_utc": (self.now + timedelta(hours=2)).isoformat(),
            "title": "Test macro release",
        }

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_missing_provider_coverage_is_unknown(self, fetch):
        fetch.return_value = self.calendar([], coverage_end_utc=None)
        self.assertEqual(main_v5_3.get_event_risk(self.days)["event_risk"], "UNKNOWN")

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_complete_coverage_with_empty_schedule_is_low(self, fetch):
        result = self.calendar([])
        fetch.return_value = result
        risk = main_v5_3.get_event_risk(self.days)
        self.assertEqual(risk["event_risk"], "LOW")
        self.assertEqual(risk["calendar_event_count"], 0)

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_high_impact_eur_usd_event_is_high(self, fetch):
        fetch.return_value = self.calendar([self.event("USD", "high")])
        risk = main_v5_3.get_event_risk(self.days)
        self.assertEqual(risk["event_risk"], "HIGH")
        self.assertEqual(len(risk["high_impact_events"]), 1)

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_medium_impact_eur_usd_event_is_medium(self, fetch):
        fetch.return_value = self.calendar([self.event("EUR", "medium")])
        risk = main_v5_3.get_event_risk(self.days)
        self.assertEqual(risk["event_risk"], "MEDIUM")
        self.assertEqual(len(risk["medium_impact_events"]), 1)

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_short_provider_coverage_is_unknown(self, fetch):
        fetch.return_value = self.calendar(
            [], coverage_end_utc=(self.now + timedelta(days=2)).isoformat()
        )
        self.assertEqual(main_v5_3.get_event_risk(self.days)["event_risk"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
