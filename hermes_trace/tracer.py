"""TraceGraph — in-memory trace graph for Hermes agent execution.

Captures the agent's execution as a directed graph:
  Session → Turns → (LLM calls → Tool calls, Subagents)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TRACE_DIR = Path.home() / ".hermes" / "traces"

# Status values that represent successful completion
_OK_STATUSES = frozenset({"completed", "ok"})


@dataclass
class Span:
    """A timed span within a trace (LLM call or tool call)."""

    name: str
    kind: str  # "llm_call", "tool_call", "subagent"
    started_at: float
    ended_at: float = 0.0
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "started"  # started, completed, error, interrupted
    children: list[Span] = field(default_factory=list)


@dataclass
class Turn:
    """One user turn — user message → LLM loop → response."""

    index: int
    user_message: str = ""
    assistant_response: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Subagent:
    """A delegated subagent."""

    child_session_id: str
    parent_session_id: str
    goal: str = ""
    status: str = "started"
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: int = 0
    summary: str = ""


@dataclass
class TraceGraph:
    """Top-level trace for one agent session."""

    session_id: str
    model: str = ""
    platform: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    turns: list[Turn] = field(default_factory=list)
    subagents: list[Subagent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Track active spans by key for matching pre/post calls
    _active_llm_span: Optional[Span] = field(default=None, repr=False, init=False)
    _active_tool_spans: dict[str, Span] = field(default_factory=dict, repr=False, init=False)
    _active_subagents: dict[str, Subagent] = field(default_factory=dict, repr=False, init=False)
    _current_turn: Optional[Turn] = field(default=None, repr=False, init=False)
    _turn_counter: int = field(default=0, repr=False, init=False)
    _span_counter: int = field(default=0, repr=False, init=False)
    _llm_counter: int = field(default=0, repr=False, init=False)  # monotonic per turn

    def ensure_started(self) -> bool:
        """Backfill started_at if the session_start hook fired before agent init.

        Returns True if a backfill was performed, False if started_at was already set.
        """
        if self.started_at == 0.0:
            self.started_at = time.time()
            logger.debug("Trace: backfilled started_at for session %s", self.session_id)
            return True
        return False

    def start_turn(self, user_message: str = "", **metadata) -> Turn:
        self._turn_counter += 1
        self._llm_counter = 0  # reset per turn
        turn = Turn(
            index=self._turn_counter,
            user_message=user_message,
            started_at=time.time(),
            metadata=metadata,
        )
        self._current_turn = turn
        self.turns.append(turn)
        logger.debug("Trace: turn %d started for session %s", self._turn_counter, self.session_id)
        return turn

    def end_turn(self, assistant_response: str = "") -> Optional[Turn]:
        turn = self._current_turn
        if turn is None:
            return None
        turn.ended_at = time.time()
        turn.assistant_response = assistant_response
        self._current_turn = None
        logger.debug("Trace: turn %d ended (%.1fs)", turn.index, turn.ended_at - turn.started_at)
        return turn

    def start_llm_call(self, **metadata) -> Span:
        self._llm_counter += 1
        # Store both the monotonic index and the hook-provided call count
        api_call_count = metadata.get("api_call_count", self._llm_counter)
        metadata["llm_index"] = self._llm_counter
        metadata.setdefault("api_call_count", api_call_count)
        span = Span(
            name="llm_call",
            kind="llm_call",
            started_at=time.time(),
            metadata=metadata,
            status="started",
        )
        self._active_llm_span = span
        if self._current_turn:
            self._current_turn.spans.append(span)
        logger.debug(
            "Trace: LLM call #%d started (hook count=%s)",
            self._llm_counter,
            api_call_count,
        )
        return span

    def end_llm_call(self, status: str = "completed", **metadata) -> Optional[Span]:
        span = self._active_llm_span
        self._active_llm_span = None
        if span is None:
            return None
        span.ended_at = time.time()
        span.duration_ms = int((span.ended_at - span.started_at) * 1000)
        span.status = status
        span.metadata.update(metadata)
        logger.debug("Trace: LLM call ended (%dms, %s)", span.duration_ms, status)
        return span

    def start_tool_call(self, tool_name: str, tool_call_id: str = "", **metadata) -> Span:
        span = Span(
            name=tool_name,
            kind="tool_call",
            started_at=time.time(),
            metadata=metadata,
            status="started",
        )
        # Use tool_call_id when available; fall back to a unique counter key
        # to prevent collisions when the same tool is called multiple times
        # in one turn without distinct IDs.
        key = tool_call_id if tool_call_id else f"{tool_name}_{self._span_counter}"
        self._span_counter += 1
        self._active_tool_spans[key] = span
        if self._current_turn:
            self._current_turn.spans.append(span)
        logger.debug("Trace: tool call '%s' started (key=%s)", tool_name, key)
        return span

    def end_tool_call(
        self, tool_name: str, tool_call_id: str = "", status: str = "completed", **metadata
    ) -> Optional[Span]:
        # Match by tool_call_id first, then fall back to scanning by tool_name
        # (handles the case where start_tool_call used a counter-based key).
        key = tool_call_id if tool_call_id else ""
        span = self._active_tool_spans.pop(key, None) if key else None
        if span is None:
            # Scan for any active span matching this tool_name
            for k, v in list(self._active_tool_spans.items()):
                if v.name == tool_name:
                    span = v
                    del self._active_tool_spans[k]
                    break
        if span is None:
            return None
        span.ended_at = time.time()
        # Prefer the hook-provided duration; fall back to wall-clock delta
        hook_duration = metadata.pop("duration_ms", None)
        if hook_duration is not None and isinstance(hook_duration, int) and hook_duration >= 0:
            span.duration_ms = hook_duration
        else:
            span.duration_ms = int((span.ended_at - span.started_at) * 1000)
        span.status = status
        span.metadata.update(metadata)
        logger.debug("Trace: tool call '%s' ended (%dms, %s)", tool_name, span.duration_ms, status)
        return span

    def start_subagent(self, child_session_id: str, goal: str = "", **metadata) -> Subagent:
        sub = Subagent(
            child_session_id=child_session_id,
            parent_session_id=self.session_id,
            goal=goal,
            status="started",
            started_at=time.time(),
        )
        self._active_subagents[child_session_id] = sub
        self.subagents.append(sub)
        logger.debug("Trace: subagent %s started", child_session_id)
        return sub

    def end_subagent(
        self, child_session_id: str, status: str = "completed", **metadata
    ) -> Optional[Subagent]:
        sub = self._active_subagents.pop(child_session_id, None)
        if sub is None:
            # Fallback: scan for matching subagent (handles out-of-order stop)
            for s in self.subagents:
                if s.child_session_id == child_session_id and s.status == "started":
                    sub = s
                    break
            else:
                return None
        sub.ended_at = time.time()
        sub.duration_ms = int((sub.ended_at - sub.started_at) * 1000)
        sub.status = status
        sub.summary = metadata.get("summary", "")
        logger.debug("Trace: subagent %s ended (%dms, %s)", child_session_id, sub.duration_ms, status)
        return sub

    def finalize(self, completed: bool = True, interrupted: bool = False):
        """Called at session end to close any dangling spans and write output."""
        self.ended_at = time.time()
        # Close any dangling LLM span
        if self._active_llm_span:
            self.end_llm_call(status="interrupted" if interrupted else "abandoned")
        # Close any dangling tool spans
        for span in list(self._active_tool_spans.values()):
            span.ended_at = time.time()
            span.duration_ms = int((span.ended_at - span.started_at) * 1000)
            span.status = "interrupted" if interrupted else "abandoned"
        self._active_tool_spans.clear()
        # Close any dangling subagents
        for sub in list(self._active_subagents.values()):
            sub.ended_at = time.time()
            sub.duration_ms = int((sub.ended_at - sub.started_at) * 1000)
            sub.status = "interrupted" if interrupted else "abandoned"
        self._active_subagents.clear()
        if self._current_turn:
            self.end_turn("[interrupted]" if interrupted else "[incomplete]")

    # ---- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire trace graph to a JSON-compatible dict."""

        def span_to_dict(s: Span) -> dict:
            return {
                "name": s.name,
                "kind": s.kind,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "metadata": s.metadata,
                "children": [span_to_dict(c) for c in s.children],
            }

        return {
            "session_id": self.session_id,
            "model": self.model,
            "platform": self.platform,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.ended_at - self.started_at, 3) if self.ended_at else 0,
            "turns": [
                {
                    "index": t.index,
                    "user_message": t.user_message[:500],
                    "assistant_response": t.assistant_response[:500],
                    "started_at": t.started_at,
                    "ended_at": t.ended_at,
                    "duration_s": round(t.ended_at - t.started_at, 3) if t.ended_at else 0,
                    "spans": [span_to_dict(s) for s in t.spans],
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "subagents": [
                {
                    "child_session_id": s.child_session_id,
                    "parent_session_id": s.parent_session_id,
                    "goal": s.goal,
                    "status": s.status,
                    "started_at": s.started_at,
                    "ended_at": s.ended_at,
                    "duration_ms": s.duration_ms,
                    "summary": s.summary[:500] if s.summary else "",
                }
                for s in self.subagents
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceGraph":
        """Reconstruct a TraceGraph from its JSON-serialized dict.

        Restores the full object tree so that to_text_tree() and other
        output methods work on loaded traces.
        """

        def dict_to_span(d: dict) -> Span:
            return Span(
                name=d.get("name", ""),
                kind=d.get("kind", ""),
                started_at=d.get("started_at", 0.0),
                ended_at=d.get("ended_at", 0.0),
                duration_ms=d.get("duration_ms", 0),
                metadata=d.get("metadata", {}),
                status=d.get("status", "completed"),
                children=[dict_to_span(c) for c in d.get("children", [])],
            )

        trace = cls(session_id=data["session_id"])
        trace.model = data.get("model", "")
        trace.platform = data.get("platform", "")
        trace.started_at = data.get("started_at", 0.0)
        trace.ended_at = data.get("ended_at", 0.0)
        trace.metadata = data.get("metadata", {})

        for t_data in data.get("turns", []):
            turn = Turn(
                index=t_data.get("index", 0),
                user_message=t_data.get("user_message", ""),
                assistant_response=t_data.get("assistant_response", ""),
                started_at=t_data.get("started_at", 0.0),
                ended_at=t_data.get("ended_at", 0.0),
                spans=[dict_to_span(s) for s in t_data.get("spans", [])],
                metadata=t_data.get("metadata", {}),
            )
            trace.turns.append(turn)
            if turn.index > trace._turn_counter:
                trace._turn_counter = turn.index

        for s_data in data.get("subagents", []):
            sub = Subagent(
                child_session_id=s_data.get("child_session_id", ""),
                parent_session_id=s_data.get("parent_session_id", ""),
                goal=s_data.get("goal", ""),
                status=s_data.get("status", "completed"),
                started_at=s_data.get("started_at", 0.0),
                ended_at=s_data.get("ended_at", 0.0),
                duration_ms=s_data.get("duration_ms", 0),
                summary=s_data.get("summary", ""),
            )
            trace.subagents.append(sub)

        return trace

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write_json(self, path: Optional[Path] = None) -> Path:
        """Write the trace as JSON to disk. Returns the output path."""
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = TRACE_DIR / f"{self.session_id}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Trace written to %s", path)
        return path

    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart of the trace."""
        def _sanitize(text: str) -> str:
            """Escape text for safe inclusion in Mermaid node labels."""
            # Replace characters that break Mermaid syntax inside quoted labels
            return (
                text.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("[", "&#91;")
                    .replace("]", "&#93;")
                    .replace("(", "&#40;")
                    .replace(")", "&#41;")
                    .replace("{", "&#123;")
                    .replace("}", "&#125;")
                    .replace("<", "&#60;")
                    .replace(">", "&#62;")
            )

        lines = ["flowchart TD"]
        lines.append(f'  session["Session: {_sanitize(self.session_id[:12])}..."]')
        lines.append(f'  session_info["model: {_sanitize(self.model)}<br/>platform: {_sanitize(self.platform)}"]')
        lines.append("  session --> session_info")

        for i, turn in enumerate(self.turns):
            tid = f"T{i}"
            turn_label = f"Turn {turn.index}"
            if turn.user_message:
                turn_label += f"<br/>user: {_sanitize(turn.user_message[:60])}"
            if turn.assistant_response:
                turn_label += f"<br/>resp: {_sanitize(turn.assistant_response[:60])}"
            lines.append(f'  {tid}["{turn_label}"]')
            lines.append(f"  session --> {tid}")

            # Show spans within the turn
            for j, span in enumerate(turn.spans):
                sid = f"{tid}_S{j}"
                if span.kind == "llm_call":
                    label = f"LLM #{span.metadata.get('llm_index','?')}"
                    label += f"<br/>{span.duration_ms}ms"
                    label += f"<br/>tokens: {span.metadata.get('usage',{}).get('input_tokens','?')}→{span.metadata.get('usage',{}).get('output_tokens','?')}"
                    lines.append(f'  {sid}["{label}"]')
                    lines.append(f"  {tid} --> {sid}")
                elif span.kind == "tool_call":
                    status_icon = "&#10003;" if span.status in _OK_STATUSES else "&#10007;"
                    label = f"{status_icon} {_sanitize(span.name)}"
                    label += f"<br/>{span.duration_ms}ms"
                    args = span.metadata.get("args", {})
                    if args:
                        args_str = _sanitize(json.dumps(args)[:80])
                        label += f"<br/>{args_str}"
                    lines.append(f'  {sid}["{label}"]')
                    lines.append(f"  {tid} --> {sid}")

        # Subagents
        for i, sub in enumerate(self.subagents):
            sid = f"SUB{i}"
            status_icon = "&#10003;" if sub.status in _OK_STATUSES else "&#10007;"
            label = f"{status_icon} subagent {_sanitize(sub.child_session_id[:12])}..."
            label += f"<br/>{sub.duration_ms}ms"
            if sub.goal:
                label += f"<br/>goal: {_sanitize(sub.goal[:60])}"
            lines.append(f'  {sid}["{label}"]')
            lines.append(f"  session --> {sid}")

        return "\n".join(lines)

    def write_mermaid(self, path: Optional[Path] = None) -> Path:
        """Write the trace as a Mermaid diagram to disk."""
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = TRACE_DIR / f"{self.session_id}.mmd"
        path.write_text(self.to_mermaid(), encoding="utf-8")
        logger.info("Mermaid trace written to %s", path)
        return path

    # ---- Stats ------------------------------------------------------------

    def compute_stats(self) -> dict[str, Any]:
        """Aggregate statistics for the entire trace.

        Returns a dict with keys:
          turns, llm_calls, tool_calls, errors,
          total_input_tokens, total_output_tokens,
          slowest_llm, slowest_tool.
        """
        llm_calls = 0
        tool_calls = 0
        errors = 0
        total_in = 0
        total_out = 0
        slowest_llm: dict[str, Any] = {}
        slowest_tool: dict[str, Any] = {}

        for turn in self.turns:
            for span in turn.spans:
                if span.kind == "llm_call":
                    llm_calls += 1
                    if span.status not in _OK_STATUSES:
                        errors += 1
                    usage = span.metadata.get("usage", {})
                    in_tok = usage.get("input_tokens", 0) or 0
                    out_tok = usage.get("output_tokens", 0) or 0
                    total_in += in_tok
                    total_out += out_tok

                    if span.duration_ms > slowest_llm.get("duration_ms", 0):
                        slowest_llm = {
                            "api_call_count": span.metadata.get("llm_index", "?"),
                            "duration_ms": span.duration_ms,
                        }

                elif span.kind == "tool_call":
                    tool_calls += 1
                    if span.status not in _OK_STATUSES:
                        errors += 1

                    if span.duration_ms > slowest_tool.get("duration_ms", 0):
                        slowest_tool = {
                            "name": span.name,
                            "duration_ms": span.duration_ms,
                        }

        return {
            "turns": len(self.turns),
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "errors": errors,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "slowest_llm": slowest_llm,
            "slowest_tool": slowest_tool,
        }

    def to_text_tree(self) -> str:
        """Generate a simple text tree of the trace (for /trace command)."""
        lines = []
        duration = round(self.ended_at - self.started_at, 1) if self.ended_at else "?"
        lines.append(f"Trace: {self.session_id}")
        lines.append(f"├── Model: {self.model} | Platform: {self.platform} | Duration: {duration}s")
        lines.append(f"├── Turns: {len(self.turns)}")
        for turn in self.turns:
            td = round(turn.ended_at - turn.started_at, 1) if turn.ended_at else "?"
            msg_preview = (turn.user_message or "")[:80]
            lines.append(f"│   ├── Turn {turn.index} ({td}s)")
            lines.append(f"│   │   ├── User: {msg_preview}")
            for j, span in enumerate(turn.spans):
                is_last_span = j == len(turn.spans) - 1 and not self.subagents
                prefix = "│   │   └──" if is_last_span else "│   │   ├──"
                if span.kind == "llm_call":
                    tokens = span.metadata.get("usage", {})
                    lines.append(
                        f"{prefix} LLM #{span.metadata.get('llm_index','?')} "
                        f"({span.duration_ms}ms, "
                        f"in:{tokens.get('input_tokens','?')} out:{tokens.get('output_tokens','?')})"
                    )
                elif span.kind == "tool_call":
                    status = "✓" if span.status in _OK_STATUSES else "✗"
                    lines.append(
                        f"{prefix} {status} {span.name} ({span.duration_ms}ms)"
                    )
            resp_preview = (turn.assistant_response or "")[:120]
            if resp_preview:
                lines.append(f"│   │   └── Response: {resp_preview}")

        if self.subagents:
            lines.append(f"├── Subagents: {len(self.subagents)}")
            for sub in self.subagents:
                status = "✓" if sub.status in _OK_STATUSES else "✗"
                lines.append(
                    f"│   ├── {status} {sub.child_session_id[:12]}... "
                    f"({sub.duration_ms}ms)"
                )

        # ---- Stats footer -------------------------------------------------

        stats = self.compute_stats()
        lines.append("")
        lines.append("─── Stats ───")

        parts = [
            f"Turns: {stats['turns']}",
            f"LLM calls: {stats['llm_calls']}",
            f"Tool calls: {stats['tool_calls']}",
        ]
        if stats["errors"]:
            parts.append(f"Errors: {stats['errors']}")
        lines.append("  ".join(parts))

        tok_parts = [
            f"Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out",
        ]
        lines.append("  ".join(tok_parts))

        slow = stats["slowest_llm"]
        if slow:
            lines.append(
                f"Slowest LLM: #{slow['api_call_count']} "
                f"({slow['duration_ms'] / 1000:.1f}s)"
            )
        slow = stats["slowest_tool"]
        if slow:
            lines.append(
                f"Slowest tool: {slow['name']} "
                f"({slow['duration_ms'] / 1000:.1f}s)"
            )

        return "\n".join(lines)

    def to_gantt(self, bar_width: int = 70) -> str:
        """ASCII Gantt chart showing span concurrency and bottlenecks.

        Each turn gets a timeline row with horizontal bars positioned
        relative to the turn's wall-clock start.  Tool spans are
        indented under their preceding LLM span.

        The slowest LLM span is annotated with ``← bottleneck``.
        """
        lines: list[str] = []
        stats = self.compute_stats()
        slowest_llm_key: tuple[int, int] | None = None
        if stats["slowest_llm"]:
            slowest_llm_key = (
                stats["slowest_llm"].get("api_call_count", -1),
                stats["slowest_llm"].get("duration_ms", 0),
            )

        for turn in self.turns:
            if not turn.spans:
                continue
            turn_dur = turn.ended_at - turn.started_at if turn.ended_at else 0
            if turn_dur <= 0:
                turn_dur = max(
                    (s.ended_at - turn.started_at)
                    for s in turn.spans
                    if s.ended_at > 0
                ) or 1

            td = round(turn_dur, 1)
            lines.append(f"Turn {turn.index} ({td}s)")

            for span in turn.spans:
                offset = max(span.started_at - turn.started_at, 0)
                dur_s = span.duration_ms / 1000

                # Build label
                if span.kind == "llm_call":
                    label = f"  LLM #{span.metadata.get('llm_index', '?')}"
                else:
                    label = f"    {span.name}"

                # Build bar
                pos = int((offset / turn_dur) * bar_width)
                width = max(1, int((dur_s / turn_dur) * bar_width))
                bar = "░" * pos + "█" * width + "░" * max(0, bar_width - pos - width)

                # Annotate bottleneck
                tag = ""
                if (
                    span.kind == "llm_call"
                    and slowest_llm_key
                    and span.metadata.get("llm_index") == slowest_llm_key[0]
                ):
                    tag = "  ← bottleneck"

                lines.append(f"{label:<20s} {bar} {dur_s:.1f}s{tag}")

        return "\n".join(lines) if lines else "(no spans)"


# Thread-safe registry of active traces by session_id
_traces: dict[str, TraceGraph] = {}
_lock = threading.RLock()


def get_trace(session_id: str) -> TraceGraph:
    """Get or create a trace for the given session."""
    with _lock:
        if session_id not in _traces:
            _traces[session_id] = TraceGraph(session_id=session_id)
        return _traces[session_id]


def remove_trace(session_id: str) -> Optional[TraceGraph]:
    """Remove and return the trace for a session."""
    with _lock:
        return _traces.pop(session_id, None)


def get_current_trace(session_id: Optional[str] = None) -> Optional[TraceGraph]:
    """Get the current active trace, optionally for a specific session."""
    with _lock:
        if session_id:
            return _traces.get(session_id)
        # Return the most recently created trace
        if _traces:
            return list(_traces.values())[-1]
        return None


def list_traces() -> list[str]:
    """List all active trace session IDs."""
    with _lock:
        return list(_traces.keys())
