# Hermes Trace

A Hermes Agent plugin that builds **execution trace graphs** — capturing every
turn, LLM call, tool call, subagent spawn, and approval request as a structured
directed graph.  View traces interactively with `/trace`, export them as JSON
or Mermaid diagrams, or query them from the terminal with `hermes trace`.

## Features

- **18 lifecycle hooks** — automatic tracing of every agent event: sessions,
  turns, API requests, tool calls, subagents, approvals, gateway dispatch,
  and result/output transformations.
- **`/trace` slash command** — view the current session's trace as a text tree,
  list active/saved traces, load past traces, or export to JSON/Mermaid.
- **`hermes trace` CLI** — query traces outside a session:

  ```
  hermes trace list              # table of saved traces
  hermes trace view <id>         # text tree
  hermes trace stats <id>        # aggregate statistics
  hermes trace clean --keep 20   # rotate old traces
  ```

- **Text tree** — human-readable tree with turn messages, LLM/tool durations,
  token counts, and success/error markers.
- **Stats footer** — totals for turns, LLM calls, tool calls, errors, tokens,
  and slowest span by type.
- **Three output formats**, auto-written to `~/.hermes/traces/` at session end:
  - **JSON** (`<session_id>.json`) — full machine-readable graph
  - **Mermaid** (`<session_id>.mmd`) — flowchart for embedding in docs/issues
  - **Text tree** — rendered on demand via `/trace` or `hermes trace view`

- **Zero runtime dependencies** — pure Python stdlib.

## Installation

```bash
# Copy the plugin into your Hermes profile
cp -r hermes_trace/ plugin.yaml ~/.hermes/plugins/hermes-trace/

# Or symlink for development
ln -s "$PWD" ~/.hermes/plugins/hermes-trace
```

Then restart Hermes or run `hermes plugins reload`.

## Architecture

```
Session
  └── Turn 1
  │     ├── User message
  │     ├── LLM call #1 (model, provider, tokens, duration)
  │     ├── Tool call: read_file (args, result, duration)
  │     ├── LLM call #2
  │     └── Assistant response
  ├── Turn 2
  │     └── ...
  └── Subagents
        └── child_session_id (goal, status, duration)
```

### Source files

| File | Purpose |
|------|---------|
| `plugin.yaml` | Plugin manifest — name, version, 18 hooks |
| `hermes_trace/__init__.py` | `register()` — hooks, slash command, CLI registration |
| `hermes_trace/tracer.py` | `TraceGraph`, `Span`, `Turn`, `Subagent` dataclasses + serialisation |
| `hermes_trace/cli.py` | `hermes trace` subcommand handlers (list, view, stats, clean) |

### Hooks

| Hook | What it captures |
|------|-----------------|
| `on_session_start` | Model, platform, start time |
| `on_session_end` | Finalise + write JSON/Mermaid |
| `on_session_finalize` | Cleanup in-memory trace |
| `on_session_reset` | Record `/new` or `/reset` events |
| `pre_llm_call` | Start a new turn, backfill `started_at` |
| `post_llm_call` | End turn, store assistant response |
| `pre_api_request` | LLM call start — model, provider, tokens, backfill session metadata |
| `post_api_request` | LLM call end — usage, finish reason, duration |
| `api_request_error` | Mark LLM call as errored |
| `pre_tool_call` | Tool call start — name, args, ID |
| `post_tool_call` | Tool call end — result, duration, status |
| `subagent_start` | Delegated subagent start — goal |
| `subagent_stop` | Subagent end — status, summary, duration |
| `pre_approval_request` | Approval prompt shown to user |
| `post_approval_response` | User's approval choice |
| `transform_tool_result` | Result seen by the model after plugin transforms |
| `transform_llm_output` | Final response after plugin transforms |
| `pre_gateway_dispatch` | Inbound gateway message (logged) |

## Usage

### In a session

```
/trace                 # show current trace as text tree
/trace view <id>       # view a specific session
/trace list            # list active traces
/trace load <id>       # load and display a saved trace
/trace export          # write JSON + Mermaid to ~/.hermes/traces/
/trace mermaid         # output Mermaid flowchart
/trace clear           # remove in-memory trace
```

### From the terminal

```
hermes trace list                    # table of all saved traces
hermes trace view 20260608_101004   # text tree (supports prefix matching)
hermes trace stats 20260608_101004  # aggregate statistics
hermes trace clean --keep 20        # keep the 20 most recent, delete the rest
```

### Output files

Traces are saved to `~/.hermes/traces/`:

```
~/.hermes/traces/
├── 20260609_203712_a15113.json   # full trace graph
├── 20260609_203712_a15113.mmd    # Mermaid flowchart
└── ...
```

## Example trace

```
Trace: 20260609_203712_a15113
├── Model: deepseek-v4-pro | Platform: cli | Duration: 720.9s
├── Turns: 5
│   ├── Turn 1 (39.0s)
│   │   ├── User: on va travailler sur hermes-trace...
│   │   ├── LLM #1 (5359ms, in:13146 out:157)
│   │   ├── ✓ read_file (58ms)
│   │   ├── ✓ search_files (39ms)
│   │   └── LLM #5 (15949ms, in:1317 out:774)
│   ├── Turn 2 (157.8s)
│   │   ├── ✓ patch (323ms)
│   │   ├── ✗ execute_code (300004ms)
│   │   └── ...
│   ...

─── Stats ───
Turns: 5  LLM calls: 28  Tool calls: 22
Tokens: 48,250 in / 7,894 out
Slowest LLM: #19 (71.8s)
Slowest tool: execute_code (300.0s)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint
ruff check hermes_trace/

# Format
ruff format hermes_trace/
```

Python 3.12+ required.  Zero runtime dependencies — only `pytest` and `ruff` for
development.

## License

MIT
