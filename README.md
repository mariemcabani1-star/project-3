# Calendar Agent

The active application is a CLI backed by a LangGraph workflow. Event creation
uses the isolated JSON calendar in `sandbox/`; the active workflow does not call
the Google Calendar integration.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
```

The CLI validates the event, displays the exact sandbox operation for approval,
and offers **yes**, **no**, or **edit**. Editing returns to approval with the
updated proposal. A unique LangGraph thread ID is also used as the request ID
for sandbox idempotency.

## Workflow responsibilities

| Component | Responsibility |
|---|---|
| `main.py` | Collect event details and handle CLI approval/edit responses |
| `agent/graph.py` | LangGraph routing and in-memory interrupt checkpointing |
| `agent/nodes.py` | Approval, edit, cancellation, and approved creation |
| `agent/state.py` | Shared workflow state |
| `agent/sandbox_adapter.py` | Validate sandbox restrictions, convert time zones, and prepare the idempotency key |
| `agent/tools/` | Tool schemas, validation, registry, and sandbox tool execution |
| `sandbox/` | Isolated JSON calendar, seed data, and reset support |
| `schemas/` | Application-level event validation |

`create_event` is the only write tool in the current graph. The registry rejects
direct write execution; the graph creation node supplies a one-use grant bound
to the exact operation only after explicit approval. Any future write tool must
also be routed through an approval checkpoint.

Invitations are recorded in the sandbox JSON outbox. No real invitations or
Google Calendar writes are made by this workflow.

## Sandbox and tests

Reset the local sandbox calendar to its seed data:

```powershell
python -m sandbox.reset
```

Run the evaluation suite (it resets a disposable sandbox before testing and
does not touch live Google Calendar):

```powershell
python -m tests.run_evaluation
```

The root-level Google integration scripts are manual and are not part of this
evaluation command.
