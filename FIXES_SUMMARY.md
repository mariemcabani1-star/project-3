# Calendar sandbox and LangGraph fixes

## Proposed commit message

```text
feat: integrate isolated calendar sandbox with LangGraph HITL

- Add a sandbox adapter to convert EventPayload values into validated tool arguments.
- Replace live Google Calendar creation in the workflow with sandbox registry execution.
- Extend CalendarState with a request ID and the exact approved operation.
- Keep application validation separate from sandbox-specific limits.
- Add seed data and reset support for reproducible sandbox initialization.
- Preserve LangGraph approval, rejection, edit, and re-approval behavior.
- Use the LangGraph thread ID as the request ID for idempotent creation.
- Preserve descriptions through approval, storage, and idempotency checks.
- Add workflow and sandbox tests for approval, edits, idempotency, and descriptions.
```

## Mistakes addressed

- The CLI now displays the approval `proposal` payload.
- The CLI creates one `thread_id`, uses it in the initial state as `request_id`,
  and resumes interruptions with `Command(resume=...)`.
- Edited events return to the approval checkpoint before they can be created.
- The graph executes the exact operation presented for approval.
- Descriptions are retained in the proposed operation and sandbox event.
- The adapter prepares sandbox operations; the creation node invokes the
  registry only after the graph's explicit approval route.
- The active LangGraph workflow does not import or call Google Calendar.

## Verification

- `python -m unittest discover -s tests -t .` passed (28 tests).
- An approved CLI event was created in a disposable sandbox; the event and
  invitation outbox entry were verified.

## Next implementation step (not implemented yet)

Add a non-breaking **Find available time** path to the existing LangGraph CLI:

1. Collect a date range and meeting duration from the user.
2. Add a read-only graph node that calls `get_calendar` and
   `find_free_slots` through the sandbox registry.
3. Present available slots and let the user choose one before collecting the
   remaining event details.
4. Route the chosen slot through the existing proposal and approval nodes.
   The write must still require explicit approval and execute the exact
   approved operation.
5. Add tests for available slots, no available slots, conflicts, and approval
   and rejection. Confirm read-only nodes never call `create_event`.

Keep the current direct-time event flow available. Keep invitations sandboxed
in the JSON outbox; real Google Calendar writes or invitation sending are out
of scope for this step.

## Remaining scope and limitations

- The graph does not yet call `get_calendar` or `find_free_slots`; it creates a
  user-proposed event rather than searching for an available time.
- Sandbox invitations are queued in the JSON outbox; no real invitations are
  sent, and there is no separate `update_event` tool.
- The approval gate applies to the LangGraph workflow path. Any future code
  that calls the registry directly must enforce its own approval gate.
- `tests/test_create_event.py` is empty. The root-level Google and LangSmith
  scripts are separate manual checks; they are not part of the documented
  unittest command.
- The `feat/langgraph-sandbox` branch currently has no commits; commit the
  reviewed project files before pushing.
