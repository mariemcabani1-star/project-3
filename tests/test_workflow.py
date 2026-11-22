from __future__ import annotations

import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.types import Command

import main
from agent.config import AgentSettings
from agent.graph import build_graph
from agent.model import ModelResponse
from agent.offline_demo import OfflineDemoModel
from agent.sandbox_adapter import prepare_sandbox_event
from agent.tools.calendar_tools import CREATE_EVENT
from agent.tools.registry import openai_specs
from sandbox import store
from schemas.event_schema import EventPayload
from tests.agent_fakes import ScriptedModel, final_step, tool_step


def create_args(title: str = "Project meeting") -> dict:
    return {
        "title": title,
        "start": "2026-12-07T14:00",
        "end": "2026-12-07T14:30",
        "attendees": ["sam@example.com"],
        "description": "Planning",
    }


def initial_state(request_id: str = "workflow-test") -> dict:
    from agent.nodes import SYSTEM_PROMPT

    return {
        "request_id": request_id,
        "user_request": "Create a project meeting.",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Create a project meeting."},
        ],
        "pending_tool_call": None,
        "approved_tool_call": None,
        "approval": "pending",
        "approval_decisions": [],
        "successful_reads": 0,
        "model_turns": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "started_at": time.monotonic(),
        "trajectory": [{"type": "user_request", "text": "Create a project meeting."}],
        "status": "running",
    }

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory(
            prefix="calendar-agent-workflow-tests-"
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

    def setUp(self):
        store.reset()

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
                                "proposal": {
                                    "title": "Project meeting",
                                    "start_time": "2026-12-07T14:00",
                                    "end_time": "2026-12-07T14:30",
                                    "attendees": ["sam@example.com"],
                                },
                            })
                        ]
                    }
                return {"result": "created"}

        graph = FakeGraph()
        with (
            patch("main.build_graph", return_value=graph),
            patch("builtins.input", return_value="1"),
            patch("main.display_proposal"),
        ):
            main.run_workflow(
                "Create a project meeting.",
                model_client=ScriptedModel([]),
            )

        state, config = graph.calls[0]
        self.assertEqual(
            state["request_id"],
            config["configurable"]["thread_id"],
        )
        self.assertEqual(
            state["messages"][-1],
            {"role": "user", "content": "Create a project meeting."},
        )
        resume_command, resumed_config = graph.calls[1]
        self.assertIsInstance(resume_command, Command)
        self.assertEqual(resume_command.resume, {"decision": "yes"})
        self.assertEqual(resumed_config, config)

    def test_live_cli_requires_explicit_mode_and_uses_dotenv_values(self):
        with patch("sys.argv", ["main.py"]):
            with self.assertRaises(SystemExit):
                main.main()

        with (
            patch("sys.argv", ["main.py", "--mode", "live"]),
            patch("main.load_dotenv") as load_dotenv,
            patch("main.collect_user_request", return_value="Find a meeting time."),
            patch("main.OpenAIChatModel", return_value="live-client"),
            patch("main.run_workflow", return_value={"status": "completed"}) as run,
        ):
            self.assertEqual(main.main(), 0)

        load_dotenv.assert_called_once_with(override=True)
        self.assertEqual(
            run.call_args.kwargs["model_client"],
            "live-client",
        )

    def test_interactive_offline_cli_runs_read_approval_and_sandbox_write(self):
        inputs = [
            "Schedule a project meeting.",
            "Operator demo",
            "2026-12-03",
            "14:00",
            "14:30",
            "sam@example.com",
            "Offline demonstration",
            "1",
            "1",
        ]
        with (
            patch("sys.argv", ["main.py", "--mode", "offline"]),
            patch("builtins.input", side_effect=inputs),
            patch("main.load_dotenv") as load_dotenv,
            patch("main.OpenAIChatModel") as openai_model,
            patch(
                "main._traced_run_workflow",
                side_effect=AssertionError("offline mode must not trace"),
            ),
            redirect_stdout(StringIO()) as output,
        ):
            self.assertEqual(main.main(), 0)

        load_dotenv.assert_not_called()
        openai_model.assert_not_called()
        self.assertIn("0 OpenAI calls", output.getvalue())
        created = [
            event for event in store.load()["events"]
            if event["title"] == "Operator demo"
        ]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["start"], "2026-12-03T14:00")
        self.assertEqual(created[0]["attendees"], ["sam@example.com"])
        self.assertEqual(
            store.load()["outbox"],
            [{
                "to": "sam@example.com",
                "event_id": created[0]["id"],
                "type": "invite",
            }],
        )

    def test_interactive_offline_demo_does_not_propose_a_conflicting_write(self):
        event = {
            "title": "Conflict demo",
            "start": "2026-12-03T10:00",
            "end": "2026-12-03T10:30",
            "attendees": [],
            "description": "",
        }
        with redirect_stdout(StringIO()):
            result = main.run_workflow(
                "Schedule a conflicting demo.",
                model_client=OfflineDemoModel(event),
                settings=AgentSettings(),
            )

        self.assertEqual(result["status"], "completed")
        self.assertIn("conflicts", result["result"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["approval_decisions"], [])
        self.assertEqual(len(store.load()["events"]), 8)

    def test_interactive_offline_cli_rejection_does_not_create_event(self):
        event = {
            "title": "Rejected demo",
            "start": "2026-12-03T14:00",
            "end": "2026-12-03T14:30",
            "attendees": [],
            "description": "",
        }
        with (
            patch("builtins.input", return_value="2"),
            redirect_stdout(StringIO()),
        ):
            result = main.run_workflow(
                "Schedule a rejected demo.",
                model_client=OfflineDemoModel(event),
                settings=AgentSettings(),
            )

        self.assertEqual(result["status"], "completed")
        self.assertIn("rejected", result["result"].lower())
        self.assertEqual(len(store.load()["events"]), 8)

    def test_read_observation_returns_to_model_before_final_answer(self):
        model = ScriptedModel([
            tool_step("get_calendar", {
                "start_date": "2026-12-07",
                "end_date": "2026-12-07",
            }),
            final_step("The calendar is clear."),
        ])
        graph = build_graph(model_client=model)
        result = graph.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "read-then-answer"}},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"], "The calendar is clear.")
        self.assertEqual(result["model_turns"], 2)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(
            [entry["type"] for entry in result["trajectory"]],
            ["user_request", "model_response", "tool_observation", "model_response"],
        )
        self.assertEqual(result["messages"][-2]["role"], "tool")

    def test_write_without_successful_read_is_rejected_without_approval(self):
        model = ScriptedModel([
            tool_step("create_event", create_args()),
            final_step("Read calendar availability before a write."),
        ])
        graph = build_graph(model_client=model)
        result = graph.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "write-needs-read"}},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(store.load()["events"]), 8)
        observation = result["trajectory"][2]["observation"]
        self.assertEqual(observation["error"]["code"], "availability_check_required")
        self.assertEqual(result["approval_decisions"], [])

    def test_approved_write_runs_only_after_read_and_human_confirmation(self):
        model = ScriptedModel([
            tool_step("find_free_slots", {
                "start_date": "2026-12-07",
                "end_date": "2026-12-07",
                "duration_minutes": 30,
            }),
            tool_step("create_event", create_args()),
            final_step("Created the sandbox event."),
        ])
        graph = build_graph(model_client=model)
        config = {"configurable": {"thread_id": "approved-write"}}
        result = graph.invoke(initial_state("approved-write"), config=config)
        approval = result["__interrupt__"][0].value
        self.assertEqual(approval["type"], "approval")
        self.assertEqual(approval["proposal"]["title"], "Project meeting")
        self.assertEqual(approval["proposal"]["attendees"], ["sam@example.com"])
        self.assertEqual(len(store.load()["events"]), 8)

        result = graph.invoke(Command(resume={"decision": "yes"}), config=config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["approval_decisions"], ["yes"])
        self.assertEqual(len(store.load()["events"]), 9)
        self.assertEqual(store.load()["outbox"], [])
        self.assertEqual(result["trajectory"][-2]["type"], "write_result")

    def test_sandbox_invite_requires_approval_and_queues_no_email(self):
        model = ScriptedModel([
            tool_step("get_calendar", {
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
            }),
            tool_step("send_invite", {
                "event_id": "evt_0001",
                "attendees": ["team@example.com"],
            }),
            final_step("The invitation is queued in the sandbox outbox."),
        ])
        graph = build_graph(model_client=model)
        config = {"configurable": {"thread_id": "approved-invite"}}
        result = graph.invoke(initial_state("approved-invite"), config=config)
        interrupt_value = result["__interrupt__"][0].value

        self.assertEqual(interrupt_value["type"], "approval")
        self.assertEqual(interrupt_value["tool"], "send_invite")
        self.assertEqual(interrupt_value["proposal"]["title"], "Team standup")
        self.assertEqual(
            interrupt_value["proposal"]["attendees"],
            ["team@example.com"],
        )
        self.assertEqual(store.load()["outbox"], [])

        result = graph.invoke(Command(resume={"decision": "yes"}), config=config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(store.load()["outbox"], [{
            "to": "team@example.com",
            "event_id": "evt_0001",
            "type": "invite",
        }])
        self.assertEqual(result["approval_decisions"], ["yes"])

    def test_rejected_sandbox_invite_does_not_change_outbox(self):
        model = ScriptedModel([
            tool_step("get_calendar", {
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
            }),
            tool_step("send_invite", {
                "event_id": "evt_0001",
                "attendees": ["team@example.com"],
            }),
            final_step("No invitation was queued."),
        ])
        graph = build_graph(model_client=model)
        config = {"configurable": {"thread_id": "rejected-invite"}}
        graph.invoke(initial_state("rejected-invite"), config=config)
        result = graph.invoke(Command(resume={"decision": "no"}), config=config)

        self.assertEqual(result["approval_decisions"], ["no"])
        self.assertEqual(store.load()["outbox"], [])

    def test_editing_invite_recipient_requires_new_approval(self):
        model = ScriptedModel([
            tool_step("get_calendar", {
                "start_date": "2026-12-04",
                "end_date": "2026-12-04",
            }),
            tool_step("send_invite", {
                "event_id": "evt_0008",
                "attendees": ["amine@example.com"],
            }),
            final_step("Updated invitation recipients were queued."),
        ])
        graph = build_graph(model_client=model)
        config = {"configurable": {"thread_id": "edit-invite"}}
        graph.invoke(initial_state("edit-invite"), config=config)
        edit_result = graph.invoke(
            Command(resume={"decision": "edit"}),
            config=config,
        )
        self.assertEqual(edit_result["__interrupt__"][0].value["type"], "edit")

        approval_result = graph.invoke(
            Command(resume={
                "event_id": "evt_0008",
                "attendees": ["sara@example.com"],
            }),
            config=config,
        )
        proposal = approval_result["__interrupt__"][0].value["proposal"]
        self.assertEqual(proposal["attendees"], ["sara@example.com"])
        self.assertEqual(store.load()["outbox"], [])

        result = graph.invoke(Command(resume={"decision": "yes"}), config=config)
        self.assertEqual(result["approval_decisions"], ["edit", "yes"])
        self.assertEqual(store.load()["outbox"], [{
            "to": "sara@example.com",
            "event_id": "evt_0008",
            "type": "invite",
        }])

    def test_rejection_does_not_create_event(self):
        model = ScriptedModel([
            tool_step("get_calendar", {
                "start_date": "2026-12-07",
                "end_date": "2026-12-07",
            }),
            tool_step("create_event", create_args()),
            final_step("No event was created."),
        ])
        graph = build_graph(model_client=model)
        config = {"configurable": {"thread_id": "rejected-write"}}
        graph.invoke(initial_state("rejected-write"), config=config)
        result = graph.invoke(Command(resume="no"), config=config)

        self.assertEqual(result["approval_decisions"], ["no"])
        self.assertEqual(len(store.load()["events"]), 8)
        self.assertEqual(result["trajectory"][-2]["type"], "approval_result")

    def test_edit_changes_operation_and_requires_a_new_approval(self):
        model = ScriptedModel([
            tool_step("get_calendar", {
                "start_date": "2026-12-07",
                "end_date": "2026-12-07",
            }),
            tool_step("create_event", create_args()),
            final_step("The edited event was created."),
        ])
        graph = build_graph(model_client=model)
        config = {"configurable": {"thread_id": "edit-write"}}
        graph.invoke(initial_state("edit-write"), config=config)
        edit_result = graph.invoke(
            Command(resume={"decision": "edit"}), config=config
        )
        edit_request = edit_result["__interrupt__"][0].value
        self.assertEqual(edit_request["type"], "edit")

        edited = {
            "title": "Updated project meeting",
            "start": "2026-12-08T14:00",
            "end": "2026-12-08T14:30",
            "attendees": ["sam@example.com"],
            "description": "Updated plan",
        }
        approval_result = graph.invoke(Command(resume=edited), config=config)
        next_approval = approval_result["__interrupt__"][0].value
        self.assertEqual(next_approval["type"], "approval")
        self.assertEqual(next_approval["proposal"]["title"], edited["title"])
        self.assertEqual(len(store.load()["events"]), 8)

        result = graph.invoke(Command(resume="yes"), config=config)
        self.assertEqual(result["approval_decisions"], ["edit", "yes"])
        self.assertEqual(len(store.load()["events"]), 9)
        self.assertEqual(store.load()["events"][-1]["description"], "Updated plan")

    def test_model_tool_schema_hides_system_managed_idempotency_key(self):
        model_schema = next(
            spec["function"]["parameters"]
            for spec in openai_specs()
            if spec["function"]["name"] == "create_event"
        )
        self.assertNotIn("idempotency_key", model_schema["properties"])
        self.assertNotIn("idempotency_key", model_schema["required"])
        self.assertNotIn("x-format-hint", str(model_schema))
        self.assertIn("idempotency_key", CREATE_EVENT.parameters["properties"])

    def test_application_schema_remains_separate_from_sandbox_limits(self):
        payload = EventPayload(
            title="T" * 121,
            start_time="2026-12-03T14:00:00+01:00",
            end_time="2026-12-03T15:00:00+01:00",
            attendees=[f"user{i}@example.com" for i in range(21)],
        ).model_dump(mode="json")
        self.assertEqual(len(EventPayload.model_validate(payload).attendees), 21)
        with self.assertRaisesRegex(ValueError, "titles cannot exceed 120"):
            prepare_sandbox_event(payload, "sandbox-limit-test")

    def test_live_cost_is_computed_from_configured_token_rates(self):
        settings = AgentSettings(
            input_usd_per_1m_tokens=2.0,
            output_usd_per_1m_tokens=8.0,
        )
        self.assertAlmostEqual(settings.cost_for_tokens(1000, 500), 0.006)

    def test_live_model_and_cost_ceiling_are_fixed(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "another-model"}, clear=True):
            settings = AgentSettings.from_env()
        self.assertEqual(settings.model, "gpt-4o-mini")
        self.assertEqual(settings.max_cost_usd, 0.10)
        with self.assertRaisesRegex(ValueError, "fixed to gpt-4o-mini"):
            AgentSettings(model="another-model")
        with self.assertRaisesRegex(ValueError, "cannot exceed \\$0.10"):
            AgentSettings(max_cost_usd=0.11)

    def test_live_preflight_caps_output_tokens_to_remaining_budget(self):
        class LiveSpyModel:
            is_live = True

            def __init__(self):
                self.called_with = None

            def complete(
                self,
                messages,
                tools,
                *,
                model,
                max_completion_tokens,
                allow_tools,
                timeout_seconds,
            ):
                self.called_with = {
                    "model": model,
                    "max_completion_tokens": max_completion_tokens,
                    "allow_tools": allow_tools,
                    "message_count": len(messages),
                    "tool_count": len(tools),
                    "timeout_seconds": timeout_seconds,
                }
                return ModelResponse(
                    assistant_message={"role": "assistant", "content": "Done."},
                    input_tokens=0,
                    output_tokens=1,
                    usage_source="test",
                )

        model = LiveSpyModel()
        settings = AgentSettings(
            max_cost_usd=0.0015,
            input_usd_per_1m_tokens=0,
            output_usd_per_1m_tokens=1000,
        )
        graph = build_graph(model_client=model, settings=settings)
        result = graph.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "live-budget-cap"}},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(model.called_with["model"], "gpt-4o-mini")
        self.assertEqual(model.called_with["max_completion_tokens"], 1)

    def test_live_preflight_blocks_call_when_input_reserve_exceeds_budget(self):
        class LiveSpyModel:
            is_live = True

            def __init__(self):
                self.called = False

            def complete(self, *args, **kwargs):
                self.called = bool(args or kwargs)
                raise AssertionError("Over-budget request reached the model client.")

        model = LiveSpyModel()
        settings = AgentSettings(
            max_cost_usd=0.10,
            input_usd_per_1m_tokens=1_000_000,
            output_usd_per_1m_tokens=1000,
        )
        graph = build_graph(model_client=model, settings=settings)
        result = graph.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "live-budget-block"}},
        )

        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["stop_reason"], "cost_limit")
        self.assertFalse(model.called)

    def test_model_timeout_stops_without_executing_a_tool(self):
        class TimeoutModel:
            is_live = False

            def complete(self, *args, **kwargs):
                if not args or not kwargs:
                    raise AssertionError("Graph did not pass the model request.")
                raise TimeoutError("timed out")

        graph = build_graph(model_client=TimeoutModel())
        result = graph.invoke(
            initial_state(),
            config={"configurable": {"thread_id": "model-timeout"}},
        )

        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["stop_reason"], "timeout")
        self.assertEqual(len(store.load()["events"]), 8)


if __name__ == "__main__":
    unittest.main()
