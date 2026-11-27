import ast
import unittest
from pathlib import Path


class LegacyCalendarSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path("main.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="main.py")
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_static_calendar"
        )
        isolated_module = ast.Module(body=[function], type_ignores=[])
        namespace = {}
        exec(compile(isolated_module, "main.py", "exec"), namespace)
        cls.get_calendar = staticmethod(namespace["get_static_calendar"])

    def test_unverified_calendar_is_unknown_not_low(self):
        result = self.get_calendar(3)
        self.assertEqual(result["risk"], "UNKNOWN")
        self.assertFalse(result["blocking_window_active"])

    def test_unverified_calendar_never_claims_events_were_checked(self):
        result = self.get_calendar(7)
        self.assertEqual(result["events_in_horizon"], [])
        self.assertIn("No verified economic calendar", result["reason"])


if __name__ == "__main__":
    unittest.main()
