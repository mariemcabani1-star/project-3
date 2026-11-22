
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from agent.state import CalendarState
from agent.nodes import (
    approval_node,
    edit_event_node,
    route_after_approval,
    create_event_node,
    cancel_event_node,
)

def build_graph():

    workflow = StateGraph(CalendarState)

    workflow.add_node("approval", approval_node)
    workflow.add_node("edit", edit_event_node)
    workflow.add_node("create", create_event_node)
    workflow.add_node("cancel", cancel_event_node)

    workflow.add_edge(START, "approval")

    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "create": "create",
            "edit": "edit",
            "cancel": "cancel",
        }
    )

    # After editing, approval is required again.
    workflow.add_edge("edit", "approval")

    workflow.add_edge("create", END)
    workflow.add_edge("cancel", END)

    # Keeps the interrupted workflow state during this process.
    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)