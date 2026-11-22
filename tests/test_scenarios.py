from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.run_scenarios import run_scenarios
from tests.scenarios import EXPECTED_CATEGORY_COUNTS, SCENARIOS


class ScenarioEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory(
            prefix="calendar-agent-scenario-tests-"
        )
        cls._previous_environment = {
            key: os.environ.get(key)
            for key in ("CALENDAR_SANDBOX_DIR", "TOOL_LOG_PATH")
        }
        os.environ["CALENDAR_SANDBOX_DIR"] = cls._temporary_directory.name
        os.environ["TOOL_LOG_PATH"] = str(
            Path(cls._temporary_directory.name) / "tool_calls.jsonl"
        )

    @classmethod
    def tearDownClass(cls):
        for key, value in cls._previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls._temporary_directory.cleanup()

    def test_scenario_inventory_and_trajectories(self):
        self.assertEqual(len(SCENARIOS), 36)
        report = run_scenarios()
        self.assertEqual(report["category_counts"], EXPECTED_CATEGORY_COUNTS)
        failures = [
            f"{scenario['id']}: {scenario['failure_reasons']}"
            for scenario in report["scenarios"]
            if not scenario["pass"]
        ]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
