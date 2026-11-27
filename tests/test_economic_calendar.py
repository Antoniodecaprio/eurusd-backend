import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import main_v5_3


class EconomicCalendarRiskTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.days = 7
        self.start = (self.now - timedelta(hours=6)).isoformat()
        self.end = (self.now + timedelta(days=self.days)).isoformat()

    def calendar(self, events, **overrides):
        result = {
            "status": "available",
            "provider": "ShowMyTrades",
            "server_time_utc": self.now.isoformat(),
            "coverage_start_utc": self.start,
            "coverage_end_utc": self.end,
            "events": events,
            "event_count": len(events),
        }
        result.update(overrides)
        return result

    def event(self, currency="USD", impact="high", when=None):
        when = when or (self.now + timedelta(hours=2))
        return {
            "currency": currency,
            "impact": impact,
            "datetime_utc": when.isoformat(),
            "title": "Test macro release",
        }

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_missing_provider_coverage_is_unknown(self, fetch):
        fetch.return_value = self.calendar(
            [], coverage_end_utc=None
        )
        result = main_v5_3.get_event_risk(self.days)
        self.assertEqual(result["event_risk"], "UNKNOWN")

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_complete_coverage_with_empty_schedule_is_low(self, fetch):
        fetch.return_value = self.calendar([])
        result = main_v5_3.get_event_risk(self.days)
        self.assertEqual(result["event_risk"], "LOW")
        self.assertEqual(result["calendar_event_count"], 0)

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_high_impact_eur_usd_event_is_high(self, fetch):
        fetch.return_value = self.calendar([self.event("USD", "high")])
        result = main_v5_3.get_event_risk(self.days)
        self.assertEqual(result["event_risk"], "HIGH")
        self.assertEqual(len(result["high_impact_events"]), 1)

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_medium_impact_eur_usd_event_is_medium(self, fetch):
        fetch.return_value = self.calendar([self.event("EUR", "medium")])
        result = main_v5_3.get_event_risk(self.days)
        self.assertEqual(result["event_risk"], "MEDIUM")
        self.assertEqual(len(result["medium_impact_events"]), 1)

    @patch("main_v5_3.get_showmytrades_calendar")
    def test_provider_coverage_shorter_than_requested_horizon_is_unknown(self, fetch):
        fetch.return_value = self.calendar(
            [], coverage_end_utc=(self.now + timedelta(days=2)).isoformat()
        )
        result = main_v5_3.get_event_risk(self.days)
        self.assertEqual(result["event_risk"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
