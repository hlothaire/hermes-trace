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

    def start_turn(self, user_message: str = "", **metadata) -> Turn:
        self._turn_counter += 1
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
        logger.debug("Trace: LLM call started (call #%s)", metadata.get("api_call_count", "?"))
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

