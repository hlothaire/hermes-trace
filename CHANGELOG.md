# Changelog

All notable changes to hermes-trace.

## [0.1.0] — 2026-06-15

First public release.

### Added

- **18 lifecycle hooks** for automatic tracing of all agent activity:
  session lifecycle, turns, LLM calls, tool calls, subagents, approvals,
  gateway dispatch, result and output transformations.
- **`/trace` slash command** — `view`, `list`, `load`, `export`, `mermaid`,
  `gantt`, `clear`.
- **`hermes trace` CLI** — `list`, `view`, `stats`, `gantt`, `clean`
  subcommands, usable outside a session.
- **Text tree** output with turn messages, LLM/tool durations, token counts,
  and success/error markers.
- **Stats footer** with aggregate totals: turns, LLM/tool calls, errors,
  tokens, slowest spans.
- **Gantt timeline** — ASCII bar chart showing span concurrency and
  bottlenecks per turn.
- **Subagent linking** — child traces reference their parent session
  (`parent_trace` field) for bidirectional navigation.
- **Three output formats** auto-written to `~/.hermes/traces/` at session end:
  JSON, Mermaid flowchart, text tree.

### Fixed

- `on_session_start` fires before agent initialisation → model/platform/started_at
  are now backfilled from the first API call.
- Duration showing 1.7 billion seconds → started_at is backfilled at first turn.
- `/trace load <id>` dumping 50+ KB of raw JSON → now reconstructs and displays
  the text tree.
- LLM numbering resetting after context compression → monotonic `llm_index`
  counter per turn.
- Hook-provided `duration_ms` being ignored in `end_tool_call` → now preferred
  over wall-clock delta.
- Status normalisation: `"ok"` from hooks is now treated as success (alongside
  `"completed"`), fixing false error counts.

### Changed

- All 18 hook callbacks wrapped in try/except — a single broken callback never
  orphans spans or crashes the agent.
- `README.md` promotes `hermes plugins install` as the primary installation
  method.
