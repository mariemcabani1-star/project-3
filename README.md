# Calendar Agent

The active CLI uses **LangGraph for orchestration** and OpenAI as its model
provider. The model can call the sandbox's read tools, observe their results,
and decide what to do next. Every event write returns to the graph's explicit
human approval checkpoint; the model cannot execute writes directly.

The workflow uses only the isolated JSON calendar in `sandbox/`. It does not
call Google Calendar. Invitations are queued in the sandbox outbox, not sent.

## Execution modes

### Offline evaluation (no API key or API calls)

```powershell
python -m tests.run_evaluation
python -m tests.run_scenarios
```

Both commands use scripted model clients and a disposable seeded JSON sandbox.
They do not load `.env`, import the OpenAI SDK, or make API calls. The scenario
runner writes the 36-trajectory report to `evaluation/results.json`.

### Live agent (requires API credits)

```powershell
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If `.env` already exists, edit it rather than replacing it. Set
`OPENAI_API_KEY` and the current input/output model prices in USD per million
tokens, then explicitly opt in:

```powershell
python main.py --mode live
```

Live mode loads `.env` with its values taking precedence over inherited shell
variables, so a different `OPENAI_API_KEY` in the VS Code terminal is not used.
The live model is fixed to `gpt-4o-mini`.

The last live attempt returned OpenAI error `429: You have no credits
remaining`. This indicates that the billing project or organization for the
active key has no available API credits; it is not a LangGraph or sandbox
failure. Live requests require a funded OpenAI project.

If you do not have credits, use offline evaluation instead. It uses scripted
model clients and does not load `.env`, import the OpenAI SDK, or make API
requests:

```powershell
python -m tests.run_evaluation
python -m tests.run_scenarios
```

The second command runs all 36 scenarios and writes their trajectories to
`evaluation/results.json`.

The CLI accepts a natural-language request. For example, ask it to find a
30-minute opening and schedule a project meeting with `sam@example.com`. The
agent reads the sandbox calendar, shows the exact title, local start/end time,
and attendees at the approval prompt, and supports **Yes**, **No**, or **Edit**.
An edit is shown for approval again before the write can run.

All sandbox tool times use local `Africa/Tunis` time in
`YYYY-MM-DDTHH:MM` format. If the requested time zone is unclear, the model
should ask for clarification.

## Execution limits

Defaults can be overridden in `.env`:

| Setting | Default |
|---|---:|
| `MAX_MODEL_TURNS` | 8 |
| `MAX_TOOL_CALLS` | 15 |
| `MAX_RUNTIME_SECONDS` | 60 |
| `MAX_COMPLETION_TOKENS` | 512 |

The per-request cost ceiling is fixed at `$0.10` and cannot be raised through
`.env`.

Set `OPENAI_INPUT_USD_PER_1M` and `OPENAI_OUTPUT_USD_PER_1M` to the current
prices for the selected model. Before each request, live mode reserves input
cost using a conservative serialized-request token ceiling and limits output
tokens to what remains under the budget. After each response, it calculates
cost from provider-reported token usage and the configured prices. This is an
application-side limit, not a provider-guaranteed billing ceiling.

## Workflow responsibilities

| Component | Responsibility |
|---|---|
| `main.py` | Collect a request and handle approval/edit interrupts |
| `agent/graph.py` | LangGraph model/read/write orchestration and in-memory checkpoints |
| `agent/nodes.py` | Model calls, bounds, tool observations, HITL, and approved writes |
| `agent/model.py` | OpenAI Chat Completions provider and model response protocol |
| `agent/config.py` | Model, turn, tool, runtime, token, and estimated-cost limits |
| `agent/state.py` | Shared loop state and trajectory |
| `agent/sandbox_adapter.py` | Prepare sandbox operations and request-scoped idempotency |
| `agent/tools/` | Tool schemas, validation, approval grants, and execution |
| `sandbox/` | Isolated JSON calendar, seed data, reset, and invitation outbox |
| `schemas/` | Application-level event validation |

The model-facing tool set contains `get_calendar`, `find_free_slots`, and
`create_event`. At least one successful calendar read is required before an
event write can be presented for approval. The registry rejects direct writes,
and the graph's one-use approval grant is bound to the exact operation. Future
write tools must also be routed through an explicit approval checkpoint; the
current approval form fails closed for unsupported write operations.

Tool observations are treated as untrusted input. The prompt-injection
scenarios test that even an attempted write still requires human approval;
they do not claim to prove resistance of a live language model to every attack.

## Sandbox and tests

Reset the local calendar to its known seed:

```powershell
python -m sandbox.reset
```

The offline scenarios' token counts and costs are synthetic. They test graph
routing, tools, approval, rejection, injection handling, and recovery—not live
model accuracy. Root-level Google and LangSmith scripts are separate manual
checks and are not part of the offline commands.
