import json
import os
import tempfile
import unittest

_evaluation_sandbox = os.environ.get("CALENDAR_AGENT_EVALUATION_SANDBOX")
_tmp = _evaluation_sandbox or tempfile.mkdtemp()
os.environ["CALENDAR_SANDBOX_DIR"] = _tmp
os.environ["TOOL_LOG_PATH"] = os.path.join(_tmp, "calls.jsonl")

from agent.tools import describe_call, execute, needs_approval, openai_specs, precheck  # noqa: E402
from agent.tools.registry import _issue_write_approval, execute_approved  # noqa: E402
from sandbox import store  # noqa: E402


def call(name, **args):
    if needs_approval(name):
        error = precheck(name, args)
        if error:
            return json.loads(error)
        approval = _issue_write_approval(name, args)
        return json.loads(execute_approved(name, args, approval))
    return json.loads(execute(name, args))


def new_event(**over):
    base = dict(title="Meeting", start="2026-12-01T15:00", end="2026-12-01T16:00",
                attendees=["a@example.com"], idempotency_key="key-00000001")
    base.update(over)
    return base


class ToolTests(unittest.TestCase):
    def setUp(self):
        store.reset()

    # --- reads ---
    def test_get_calendar(self):
        r = call("get_calendar", start_date="2026-12-01", end_date="2026-12-01")
        self.assertTrue(r["ok"])
        self.assertEqual(r["count"], 3)

    def test_get_calendar_empty(self):
        r = call("get_calendar", start_date="2027-01-05", end_date="2027-01-05")
        self.assertEqual(r["count"], 0)
        self.assertIn("note", r)

    def test_free_slots_avoid_busy(self):
        r = call("find_free_slots", start_date="2026-12-01", end_date="2026-12-01",
                 duration_minutes=60, max_results=20)
        starts = [s["start"] for s in r["slots"]]
        self.assertNotIn("2026-12-01T10:00", starts)       # lecture
        self.assertIn("2026-12-01T14:00", starts)

    def test_free_slots_skip_weekend(self):
        r = call("find_free_slots", start_date="2026-12-05", end_date="2026-12-06", duration_minutes=30)
        self.assertEqual(r["count"], 0)

    # --- validation / errors ---
    def test_unknown_tool(self):
        r = json.loads(execute("delete_everything", {}))
        self.assertEqual(r["error"]["code"], "unknown_tool")

    def test_bad_args_clear_message(self):
        r = call("get_calendar", start_date="tomorrow", end_date="2026-12-01")
        self.assertEqual(r["error"]["code"], "bad_arguments")
        self.assertIn("YYYY-MM-DD", r["error"]["message"])

    def test_extra_argument_rejected(self):
        r = call("get_calendar", start_date="2026-12-01", end_date="2026-12-01", delete=True)
        self.assertEqual(r["error"]["code"], "bad_arguments")

    def test_missing_argument(self):
        r = call("create_event", title="x")
        self.assertIn("missing required", r["error"]["message"])

    def test_invalid_json_string(self):
        r = json.loads(execute("get_calendar", "{not json"))
        self.assertEqual(r["error"]["code"], "bad_arguments")

    def test_impossible_date(self):
        r = call("get_calendar", start_date="2026-02-31", end_date="2026-03-01")
        self.assertEqual(r["error"]["code"], "bad_arguments")

    def test_range_too_large(self):
        r = call("get_calendar", start_date="2026-12-01", end_date="2027-02-01")
        self.assertEqual(r["error"]["code"], "range_too_large")

    # --- write tool ---
    def test_create_event_writes_and_queues_invites(self):
        r = call("create_event", **new_event())
        self.assertTrue(r["created"])
        s = store.load()
        self.assertEqual(len(s["events"]), 9)
        self.assertEqual(s["outbox"][0]["to"], "a@example.com")

    def test_registry_blocks_direct_write_without_approval(self):
        result = json.loads(execute("create_event", new_event()))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "approval_required")
        self.assertEqual(len(store.load()["events"]), 8)

    def test_approval_is_bound_to_exact_arguments_and_used_once(self):
        args = new_event()
        approval = _issue_write_approval("create_event", args)

        mismatch = json.loads(execute_approved(
            "create_event",
            {**args, "title": "Different meeting"},
            approval,
        ))
        self.assertEqual(mismatch["error"]["code"], "approval_mismatch")
        self.assertEqual(len(store.load()["events"]), 8)

        retry = json.loads(execute_approved("create_event", args, approval))
        self.assertEqual(retry["error"]["code"], "approval_reused")
        self.assertEqual(len(store.load()["events"]), 8)

    def test_description_is_stored_and_returned(self):
        r = call("create_event", **new_event(description="Discuss the launch plan"))
        self.assertEqual(r["event"]["description"], "Discuss the launch plan")

        calendar = call("get_calendar", start_date="2026-12-01", end_date="2026-12-01")
        created = next(event for event in calendar["events"] if event["id"] == "evt_0009")
        self.assertEqual(created["description"], "Discuss the launch plan")

    def test_create_is_idempotent(self):
        call("create_event", **new_event())
        r2 = call("create_event", **new_event())
        self.assertTrue(r2["ok"])
        self.assertTrue(r2["already_existed"])
        s = store.load()
        self.assertEqual(len(s["events"]), 9)   # not 10
        self.assertEqual(len(s["outbox"]), 1)   # invite not sent twice

    def test_description_is_part_of_idempotency_fingerprint(self):
        call("create_event", **new_event(description="Original description"))
        r = call("create_event", **new_event(description="Changed description"))
        self.assertEqual(r["error"]["code"], "idempotency_conflict")

    def test_key_reuse_with_different_event(self):
        call("create_event", **new_event())
        r = call("create_event", **new_event(start="2026-12-01T16:00", end="2026-12-01T17:00"))
        self.assertEqual(r["error"]["code"], "idempotency_conflict")

    def test_conflict_rejected(self):
        r = call("create_event", **new_event(start="2026-12-01T10:30", end="2026-12-01T11:30"))
        self.assertEqual(r["error"]["code"], "conflict")
        self.assertIn("find_free_slots", r["error"]["hint"])
        self.assertEqual(len(store.load()["events"]), 8)

    def test_end_before_start(self):
        r = call("create_event", **new_event(start="2026-12-01T16:00", end="2026-12-01T15:00"))
        self.assertEqual(r["error"]["code"], "bad_arguments")

    def test_bad_email(self):
        r = call("create_event", **new_event(attendees=["not-an-email"]))
        self.assertEqual(r["error"]["code"], "bad_arguments")

    def test_reset_restores_seed(self):
        call("create_event", **new_event())
        store.reset()
        self.assertEqual(len(store.load()["events"]), 8)

    # --- loop helpers ---
    def test_only_write_tool_needs_approval(self):
        self.assertTrue(needs_approval("create_event"))
        self.assertFalse(needs_approval("get_calendar"))
        self.assertFalse(needs_approval("find_free_slots"))
        self.assertIn("Create event", describe_call("create_event", new_event()))

    def test_openai_specs(self):
        specs = openai_specs()
        self.assertEqual({s["function"]["name"] for s in specs},
                         {"get_calendar", "find_free_slots", "create_event"})

    def test_calls_are_logged(self):
        call("get_calendar", start_date="2026-12-01", end_date="2026-12-01")
        with open(os.environ["TOOL_LOG_PATH"]) as f:
            rec = json.loads(f.readlines()[-1])
        self.assertEqual(rec["tool"], "get_calendar")
        self.assertIn("duration_ms", rec)

    def test_write_precheck_attempts_are_logged(self):
        error = precheck("create_event", {"title": "Missing required details"})
        self.assertIsNotNone(error)
        with open(os.environ["TOOL_LOG_PATH"]) as f:
            rec = json.loads(f.readlines()[-1])
        self.assertEqual(rec["tool"], "create_event")
        self.assertEqual(rec["phase"], "precheck")
        self.assertEqual(rec["output"]["error"]["code"], "bad_arguments")


if __name__ == "__main__":
    unittest.main()
