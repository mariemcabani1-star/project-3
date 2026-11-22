import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.types import Command

import main
from agent.graph import build_graph
from agent.sandbox_adapter import prepare_sandbox_event
from schemas.event_schema import EventPayload


def event(title="Project meeting", description="Planning"):
    return EventPayload(
        title=title,
        start_time="2026-12-03T14:00:00+01:00",
        end_time="2026-12-03T15:00:00+01:00",
        attendees=["sam@example.com"],
        description=description,
    ).model_dump(mode="json")


def interrupted_request(result):
    return result["__interrupt__"][0].value


class WorkflowTests(unittest.TestCase):
    def test_cli_uses_thread_id_as_request_id_and_resumes_approval(self):
        class FakeGraph:
            def __init__(self):
                self.calls = []

            def invoke(self, value, config):
                self.calls.append((value, config))
                if len(self.calls) == 1:
                    return {
                        "__interrupt__": [
                            SimpleNamespace(value={
                                "type": "approval",
                                "proposal": event(),
                            })
                        ]
                    }
                return {"result": "created"}

        graph = FakeGraph()
        with (
            patch("main.build_graph", return_value=graph),
            patch("builtins.input", return_value="yes"),
            patch("main.display_event"),
        ):
            main.run_workflow(event())

        initial_state, config = graph.calls[0]
        self.assertEqual(
            initial_state["request_id"],
            config["configurable"]["thread_id"],
        )
        resume_command, resumed_config = graph.calls[1]
        self.assertIsInstance(resume_command, Command)
        self.assertEqual(resume_command.resume, "yes")
        self.assertEqual(resumed_config, config)

    def test_edit_requires_reapproval_and_creates_exact_approved_proposal(self):
        graph = build_graph()
        config = {"configurable": {"thread_id": "workflow-test"}}
        result = graph.invoke(
            {
                "event": event(),
                "approval": "pending",
                "request_id": "workflow-test",
            },
            config=config,
        )
        self.assertEqual(interrupted_request(result)["proposal"]["description"], "Planning")

        result = graph.invoke(Command(resume="edit"), config=config)
        edit_request = interrupted_request(result)
        self.assertEqual(edit_request["type"], "edit")

        updated_event = event("Updated project meeting", "Updated plan")
        result = graph.invoke(Command(resume=updated_event), config=config)
        approval_request = interrupted_request(result)
        self.assertEqual(approval_request["type"], "approval")
        self.assertEqual(approval_request["proposal"]["title"], "Updated project meeting")
        self.assertEqual(approval_request["proposal"]["description"], "Updated plan")

        with patch(
            "agent.nodes.execute_approved",
            return_value=json.dumps({"ok": True, "created": True}),
        ) as execute_approved:
            result = graph.invoke(Command(resume="yes"), config=config)

        self.assertEqual(result["approval"], "completed")
        operation = execute_approved.call_args.args[1]
        self.assertEqual(operation["title"], "Updated project meeting")
        self.assertEqual(operation["description"], "Updated plan")
        self.assertEqual(operation["idempotency_key"], prepare_sandbox_event(
            updated_event, "workflow-test"
        )["idempotency_key"])

    def test_declining_never_executes_the_write_tool(self):
        graph = build_graph()
        config = {"configurable": {"thread_id": "decline-test"}}
        result = graph.invoke(
            {
                "event": event(),
                "approval": "pending",
                "request_id": "decline-test",
            },
            config=config,
        )

        with patch("agent.nodes.execute_approved") as execute:
            result = graph.invoke(Command(resume="no"), config=config)

        self.assertEqual(result["approval"], "rejected")
        execute.assert_not_called()

    def test_sandbox_limits_do_not_leak_into_application_schema(self):
        long_title = event("T" * 121)
        self.assertEqual(len(EventPayload.model_validate(long_title).title), 121)
        with self.assertRaisesRegex(ValueError, "titles cannot exceed 120"):
            prepare_sandbox_event(long_title, "long-title-test")

        many_attendees = event()
        many_attendees["attendees"] = [f"user{i}@example.com" for i in range(21)]
        self.assertEqual(len(EventPayload.model_validate(many_attendees).attendees), 21)
        with self.assertRaisesRegex(ValueError, "at most 20 attendees"):
            prepare_sandbox_event(many_attendees, "many-attendees-test")

    def test_sandbox_idempotency_requires_request_id(self):
        with self.assertRaisesRegex(ValueError, "request ID is required"):
            prepare_sandbox_event(event(), " ")

        self.assertNotEqual(
            prepare_sandbox_event(event(), "request-one")["idempotency_key"],
            prepare_sandbox_event(event(), "request-two")["idempotency_key"],
        )


if __name__ == "__main__":
    unittest.main()
