"""Fixtures for hermes-trace tests."""

import pytest
from hermes_trace.tracer import TraceGraph


@pytest.fixture
def trace() -> TraceGraph:
    """Fresh TraceGraph with started_at already set."""
    t = TraceGraph(session_id="test-session-001")
    t.model = "test-model"
    t.platform = "cli"
    t.ensure_started()
    return t


@pytest.fixture
def populated_trace(trace: TraceGraph) -> TraceGraph:
    """TraceGraph with one complete turn (LLM + 2 tool calls)."""
    trace.start_turn(user_message="hello")
    trace.start_llm_call(api_call_count=1, model="test-model", provider="test-provider")
    import time
    time.sleep(0.001)  # ensure duration_ms > 0 for stats tests
    trace.end_llm_call(
        status="completed",
        usage={"input_tokens": 100, "output_tokens": 50},
        finish_reason="stop",
        api_duration=1.2,
    )
    trace.start_tool_call(tool_name="read_file", tool_call_id="call-1", args={"path": "/tmp/x"})
    time.sleep(0.001)
    trace.end_tool_call(
        tool_name="read_file",
        tool_call_id="call-1",
        status="ok",
        result_preview="file contents",
        duration_ms=42,
    )
    trace.start_tool_call(tool_name="terminal", tool_call_id="call-2", args={"command": "ls"})
    time.sleep(0.001)
    trace.end_tool_call(
        tool_name="terminal",
        tool_call_id="call-2",
        status="error",
        result_preview="permission denied",
        duration_ms=150,
    )
    trace.end_turn(assistant_response="done")
    return trace
