# Bounded LangGraph agent and sandbox evaluation

```text
feat: add bounded OpenAI tool loop to LangGraph calendar agent

- Use LangGraph to orchestrate an OpenAI gpt-4o-mini model/read/observe loop.
- Separate offline scripted evaluation from explicitly selected live mode.
- Enforce model-turn, tool-call, runtime, completion-token, and preflight cost limits.
- Require a successful calendar read and explicit human approval before event writes.
- Keep event creation isolated to the seeded JSON sandbox with request-scoped idempotency.
- Add 36 offline trajectory scenarios covering normal use, approvals, injection, errors, recovery, and limits.
- Record per-scenario trajectories, approvals, synthetic token usage, estimated cost, and pass/fail.
```

## Implementation

- `main.py --mode live` is the only CLI path that loads `.env` or constructs the
  OpenAI client. It explicitly prioritizes project `.env` values over inherited
  shell variables and handles LangGraph approval/edit interrupts.
- `agent/graph.py` owns the model → tool → observation loop and routes every
  `create_event` proposal through the approval checkpoint.
- `agent/model.py` provides the OpenAI Chat Completions client and an injectable
  model protocol. The OpenAI SDK is imported only when the live client is called.
- `agent/config.py` supplies the default limits and requires configured model
  pricing before live requests can run.
- `agent/nodes.py` records model/tool observations, enforces bounded execution,
  checks reads before writes, and issues a one-use grant only after approval.
- Before each live request, the graph reserves a conservative input-token
  ceiling and caps output tokens using the configured per-million-token prices;
  after the request, it records actual provider-reported usage.
- Live mode fixes the model to `gpt-4o-mini` and caps per-request spending at
  `$0.10`; that cost ceiling cannot be raised through `.env`.
- `python main.py --mode live` is the live command. The offline commands are
  `python -m tests.run_evaluation` and `python -m tests.run_scenarios`; they need
  no OpenAI key or credits and do not load `.env`.
- `agent/tools/` keeps read execution in the registry and rejects direct writes.
- `tests/scenarios.py` defines exactly 36 offline scripted scenarios.
  `tests/run_scenarios.py` records calls, observations, expected and actual
  trajectories, approvals, final results, synthetic token use, estimated costs,
  and pass/fail in `evaluation/results.json`.

## Verification

Run the unit suite and offline trajectories with:

```powershell
python -m tests.run_evaluation
python -m tests.run_scenarios
```

Verified with the OpenAI SDK import-blocked and no API key set: 41 tests passed,
and all 36 scenarios passed. The detailed scenario records are in
`evaluation/results.json`.

The offline scenarios do not make OpenAI or Google API calls. Their token use
and estimated cost are synthetic; their injection cases verify the graph and
approval boundary, not the live model's ability to resist every attack.

## Remaining limitations and next step

- Calendar writes remain sandbox-only. Invitations are queued in the JSON
  outbox; no Google Calendar write or real invitation is sent.
- The cost cap is based on token estimates and provider-reported usage, not a
  provider-guaranteed maximum bill.
- The last live request returned OpenAI `429: You have no credits remaining`.
  This is an account/project billing issue, not a LangGraph or sandbox failure.
- Live model behavior has not been validated because the active OpenAI billing
  project has no credits. The offline evaluation remains runnable without them.

An OpenAI live smoke test is optional for the offline evaluation and requires
credits. When available, use a disposable JSON sandbox and current model prices
to verify a read-only request, approved creation, rejection, and edit/re-approval.
Do not enable the Google integration for that test.
