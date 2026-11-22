from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.config import AgentSettings
from agent.model import ModelClient, OpenAIChatModel
from agent.nodes import (
    approval_node,
    edit_write_node,
    execute_write_node,
    make_model_node,
    prepare_write_node,
    read_tool_node,
    route_after_approval,
    route_after_model,
)
from agent.state import CalendarState


def _route_after_preparation(state: dict[str, Any]) -> str:
    return "approval" if state.get("pending_tool_call") else "model"


def _route_after_edit(state: dict[str, Any]) -> str:
    return "approval" if state.get("pending_tool_call") else "model"


def build_graph(
    model_client: ModelClient | None = None,
    settings: AgentSettings | None = None,
):
    runtime_settings = settings or AgentSettings.from_env()
    runtime_model = model_client or OpenAIChatModel()

    workflow = StateGraph(CalendarState)
    workflow.add_node("model", make_model_node(runtime_model, runtime_settings))
    workflow.add_node("read_tool", read_tool_node)
    workflow.add_node("prepare_write", prepare_write_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("edit", edit_write_node)
    workflow.add_node(
        "write",
        lambda state: execute_write_node(state, runtime_settings),
    )

    workflow.add_edge(START, "model")
    workflow.add_conditional_edges(
        "model",
        route_after_model,
        {
            "read_tool": "read_tool",
            "prepare_write": "prepare_write",
            "end": END,
        },
    )
    workflow.add_edge("read_tool", "model")
    workflow.add_conditional_edges(
        "prepare_write",
        _route_after_preparation,
        {"approval": "approval", "model": "model"},
    )
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {"write": "write", "edit": "edit", "model": "model"},
    )
    workflow.add_conditional_edges(
        "edit",
        _route_after_edit,
        {"approval": "approval", "model": "model"},
    )
    workflow.add_edge("write", "model")

    return workflow.compile(checkpointer=InMemorySaver())
