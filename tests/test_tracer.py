"""Tests for TraceGraph — start/end pairing, cleanup, serialisation, stats."""

from __future__ import annotations

import json
import time

import pytest

from hermes_trace.tracer import (
    TRACE_DIR,
    Span,
    TraceGraph,
    Turn,
    _OK_STATUSES,
    get_trace,
    remove_trace,
)


# ---- start / end pairing --------------------------------------------------


class TestTurnPairing:
    def test_start_turn_creates_entry(self, trace):
        turn = trace.start_turn(user_message="hi")
        assert len(trace.turns) == 1
        assert turn.index == 1
        assert turn.user_message == "hi"
        assert trace._current_turn is turn

    def test_end_turn_closes_current(self, trace):
        trace.start_turn(user_message="hi")
        closed = trace.end_turn(assistant_response="hey")
        assert closed is not None
        assert closed.assistant_response == "hey"
        assert closed.ended_at > 0
        assert trace._current_turn is None

    def test_end_turn_without_start_is_noop(self, trace):
        assert trace.end_turn() is None

    def test_multiple_turns_sequential(self, trace):
        for i in range(3):
            trace.start_turn(user_message=f"msg{i}")
            trace.end_turn(assistant_response=f"resp{i}")
        assert len(trace.turns) == 3
        assert trace.turns[0].index == 1
        assert trace.turns[2].index == 3
        assert trace._current_turn is None


class TestLLMCallPairing:
    def test_start_end_llm_call(self, trace):
        trace.start_turn(user_message="hi")
        trace.start_llm_call(api_call_count=1)
        span = trace.end_llm_call(status="completed")
        assert span is not None
        assert span.kind == "llm_call"
        assert span.duration_ms >= 0  # may be 0 if start/end in same ms
        assert span.status == "completed"

    def test_end_llm_without_start_is_noop(self, trace):
        assert trace.end_llm_call() is None

    def test_llm_call_attached_to_current_turn(self, trace):
        trace.start_turn(user_message="hi")
        trace.start_llm_call(api_call_count=1)
        trace.end_llm_call(status="completed")
        trace.end_turn()
        assert len(trace.turns[0].spans) == 1
        assert trace.turns[0].spans[0].kind == "llm_call"


class TestToolCallPairing:
    def test_start_end_tool_with_id(self, trace):
        trace.start_turn(user_message="hi")
        trace.start_tool_call(tool_name="read_file", tool_call_id="abc123", args={"path": "/x"})
        span = trace.end_tool_call(tool_name="read_file", tool_call_id="abc123", status="ok")
        assert span is not None
        assert span.name == "read_file"
        assert span.kind == "tool_call"
        assert span.status == "ok"

    def test_end_tool_without_start_is_noop(self, trace):
        assert trace.end_tool_call(tool_name="read_file") is None

    def test_tool_call_attached_to_current_turn(self, trace):
        trace.start_turn(user_message="hi")
        trace.start_tool_call(tool_name="read_file", tool_call_id="t1")
        trace.end_tool_call(tool_name="read_file", tool_call_id="t1", status="ok")
        trace.end_turn()
        assert len(trace.turns[0].spans) == 1
        assert trace.turns[0].spans[0].kind == "tool_call"


# ---- counter keys / collision ----------------------------------------------


class TestCounterKeys:
    def test_same_tool_twice_without_id_does_not_collide(self, trace):
        """Two calls to the same tool in one turn without IDs should both succeed."""
        trace.start_turn(user_message="hi")
        trace.start_tool_call(tool_name="read_file")
        span1 = trace.end_tool_call(tool_name="read_file", status="ok")
        trace.start_tool_call(tool_name="read_file")
        span2 = trace.end_tool_call(tool_name="read_file", status="ok")
        trace.end_turn()
        assert span1 is not None
        assert span2 is not None
        assert span1 is not span2
        assert len(trace.turns[0].spans) == 2

    def test_matching_by_tool_name_fallback(self, trace):
        """If tool_call_id is passed at end but not at start, fall back to name."""
        trace.start_turn(user_message="hi")
        trace.start_tool_call(tool_name="terminal", args={"cmd": "ls"})
        # End with ID not used at start — should still match by name
        span = trace.end_tool_call(tool_name="terminal", tool_call_id="unused-id", status="ok")
        assert span is not None
        assert span.name == "terminal"


# ---- dangling span cleanup -------------------------------------------------


class TestDanglingCleanup:
    def test_open_llm_span_closed_on_finalize(self, trace):
        trace.start_turn(user_message="hi")
        trace.start_llm_call(api_call_count=1)
        # Don't end the LLM call
        trace.finalize()
        assert trace._active_llm_span is None
        # Span should be marked abandoned
        turn = trace.turns[0]
        assert len(turn.spans) == 1
        assert turn.spans[0].status in ("abandoned", "interrupted")

    def test_open_tool_span_closed_on_finalize(self, trace):
        trace.start_turn(user_message="hi")
        trace.start_tool_call(tool_name="read_file", tool_call_id="orphan")
        trace.finalize()
        assert not trace._active_tool_spans
        turn = trace.turns[0]
        assert len(turn.spans) == 1
        assert turn.spans[0].status in ("abandoned", "interrupted")

    def test_open_turn_closed_on_finalize(self, trace):
        trace.start_turn(user_message="hi")
        trace.finalize()
        turn = trace.turns[0]
        assert turn.ended_at > 0
        assert "[incomplete]" in turn.assistant_response or "[interrupted]" in turn.assistant_response

    def test_open_subagent_closed_on_finalize(self, trace):
        trace.start_subagent(child_session_id="child-1", goal="test")
        trace.finalize()
        assert not trace._active_subagents
        assert trace.subagents[0].status in ("abandoned", "interrupted")


# ---- ensure_started --------------------------------------------------------


class TestEnsureStarted:
    def test_backfills_when_zero(self):
        t = TraceGraph(session_id="s")
        assert t.started_at == 0.0
        result = t.ensure_started()
        assert result is True
        assert t.started_at > 0

    def test_noop_when_already_set(self, trace):
        before = trace.started_at
        result = trace.ensure_started()
        assert result is False
        assert trace.started_at == before


# ---- serialisation round-trip ----------------------------------------------


class TestSerialisation:
    def test_to_dict_has_expected_keys(self, populated_trace):
        d = populated_trace.to_dict()
        assert d["session_id"] == "test-session-001"
        assert d["model"] == "test-model"
        assert d["platform"] == "cli"
        assert len(d["turns"]) == 1
        assert d["turns"][0]["index"] == 1

    def test_json_round_trip(self, populated_trace):
        """to_dict → from_dict → to_dict should be identical."""
        original = populated_trace.to_dict()
        reconstructed = TraceGraph.from_dict(original)
        roundtripped = reconstructed.to_dict()

        # Compare keys that matter (ignore internal state)
        for key in ("session_id", "model", "platform", "turns", "subagents", "metadata"):
            assert roundtripped[key] == original[key], f"mismatch on {key}"

    def test_from_dict_empty_trace(self):
        data = {"session_id": "empty"}
        trace = TraceGraph.from_dict(data)
        assert trace.session_id == "empty"
        assert trace.turns == []
        assert trace.subagents == []

    def test_from_dict_restores_spans(self, populated_trace):
        data = populated_trace.to_dict()
        restored = TraceGraph.from_dict(data)
        assert len(restored.turns) == 1
        spans = restored.turns[0].spans
        assert len(spans) == 3  # 1 LLM + 2 tools
        kinds = [s.kind for s in spans]
        assert kinds == ["llm_call", "tool_call", "tool_call"]
        assert spans[0].metadata.get("api_call_count") == 1
        assert spans[1].name == "read_file"
        assert spans[2].name == "terminal"

    def test_to_text_tree_includes_stats(self, populated_trace):
        text = populated_trace.to_text_tree()
        assert "─── Stats ───" in text
        assert "Turns:" in text
        assert "LLM calls:" in text
        assert "Tool calls:" in text

    def test_to_mermaid_produces_flowchart(self, populated_trace):
        mmd = populated_trace.to_mermaid()
        assert mmd.startswith("flowchart TD")
        assert "Session:" in mmd

    def test_write_json_creates_file(self, populated_trace, tmp_path):
        path = tmp_path / "test.json"
        result = populated_trace.write_json(path=path)
        assert result == path
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["session_id"] == "test-session-001"


# ---- compute_stats ---------------------------------------------------------


class TestComputeStats:
    def test_empty_trace(self, trace):
        stats = trace.compute_stats()
        assert stats["turns"] == 0
        assert stats["llm_calls"] == 0
        assert stats["tool_calls"] == 0
        assert stats["errors"] == 0

    def test_populated_trace(self, populated_trace):
        stats = populated_trace.compute_stats()
        assert stats["turns"] == 1
        assert stats["llm_calls"] == 1
        assert stats["tool_calls"] == 2
        # One tool has status="error", the other "ok"
        assert stats["errors"] == 1

    def test_tokens_aggregated(self, populated_trace):
        stats = populated_trace.compute_stats()
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50

    def test_slowest_spans(self, populated_trace):
        stats = populated_trace.compute_stats()
        assert stats["slowest_llm"]["duration_ms"] > 0
        # The terminal tool had 150ms, read_file had 42ms
        assert stats["slowest_tool"]["name"] == "terminal"
        assert stats["slowest_tool"]["duration_ms"] == 150


# ---- status normalisation --------------------------------------------------


class TestStatusNormalisation:
    def test_ok_statuses(self):
        assert "completed" in _OK_STATUSES
        assert "ok" in _OK_STATUSES
        assert "error" not in _OK_STATUSES
        assert "started" not in _OK_STATUSES

    def test_ok_status_counts_as_success_in_stats(self):
        t = TraceGraph(session_id="s")
        t.ensure_started()
        t.start_turn(user_message="hi")
        t.start_tool_call(tool_name="read_file", tool_call_id="ok-call")
        t.end_tool_call(tool_name="read_file", tool_call_id="ok-call", status="ok")
        t.end_turn()
        stats = t.compute_stats()
        assert stats["errors"] == 0

    def test_error_status_counts_as_error_in_stats(self):
        t = TraceGraph(session_id="s")
        t.ensure_started()
        t.start_turn(user_message="hi")
        t.start_tool_call(tool_name="terminal", tool_call_id="err-call")
        t.end_tool_call(tool_name="terminal", tool_call_id="err-call", status="error")
        t.end_turn()
        stats = t.compute_stats()
        assert stats["errors"] == 1


# ---- edge cases ------------------------------------------------------------


class TestEdgeCases:
    def test_trace_with_no_turns_serialises(self, trace):
        d = trace.to_dict()
        assert d["turns"] == []
        assert d["subagents"] == []

    def test_multiple_turns_token_aggregation(self, trace):
        trace.start_turn(user_message="t1")
        trace.start_llm_call(api_call_count=1)
        trace.end_llm_call(status="completed", usage={"input_tokens": 10, "output_tokens": 5})
        trace.end_turn()
        trace.start_turn(user_message="t2")
        trace.start_llm_call(api_call_count=2)
        trace.end_llm_call(status="completed", usage={"input_tokens": 20, "output_tokens": 15})
        trace.end_turn()
        stats = trace.compute_stats()
        assert stats["total_input_tokens"] == 30
        assert stats["total_output_tokens"] == 20
        assert stats["llm_calls"] == 2

    def test_subagent_lifecycle(self, trace):
        trace.start_subagent(child_session_id="child-1", goal="do stuff")
        assert len(trace.subagents) == 1
        assert trace.subagents[0].status == "started"
        trace.end_subagent(child_session_id="child-1", status="completed", summary="done")
        assert trace.subagents[0].status == "completed"
        assert trace.subagents[0].summary == "done"


# ---- thread-safe registry --------------------------------------------------


class TestRegistry:
    def test_get_trace_creates_and_caches(self):
        t1 = get_trace("session-a")
        t2 = get_trace("session-a")
        assert t1 is t2
        t3 = get_trace("session-b")
        assert t1 is not t3

    def test_remove_trace(self):
        get_trace("session-x")
        removed = remove_trace("session-x")
        assert removed is not None
        assert remove_trace("session-x") is None


# ---- llm_index / compression epochs ---------------------------------------


class TestLLMIndex:
    def test_monotonic_within_turn(self, trace):
        """llm_index never resets, even when api_call_count does."""
        trace.start_turn(user_message="long turn")
        # Simulate normal calls
        trace.start_llm_call(api_call_count=1)
        trace.end_llm_call(status="completed")
        trace.start_llm_call(api_call_count=2)
        trace.end_llm_call(status="completed")
        # Simulate context compression — api_call_count resets
        trace.start_llm_call(api_call_count=1)  # hook restarted
        trace.end_llm_call(status="completed")
        trace.start_llm_call(api_call_count=2)
        trace.end_llm_call(status="completed")
        trace.end_turn()

        spans = trace.turns[0].spans
        indices = [s.metadata["llm_index"] for s in spans if s.kind == "llm_call"]
        assert indices == [1, 2, 3, 4]  # monotonic, never reset

    def test_resets_between_turns(self, trace):
        """llm_index resets to 1 for each new turn."""
        trace.start_turn(user_message="turn 1")
        trace.start_llm_call(api_call_count=1)
        trace.end_llm_call(status="completed")
        trace.start_llm_call(api_call_count=2)
        trace.end_llm_call(status="completed")
        trace.end_turn()

        trace.start_turn(user_message="turn 2")
        trace.start_llm_call(api_call_count=1)
        trace.end_llm_call(status="completed")
        trace.end_turn()

        t1_indices = [s.metadata["llm_index"] for s in trace.turns[0].spans if s.kind == "llm_call"]
        t2_indices = [s.metadata["llm_index"] for s in trace.turns[1].spans if s.kind == "llm_call"]
        assert t1_indices == [1, 2]
        assert t2_indices == [1]  # fresh start

    def test_no_api_call_count_defaults_to_llm_index(self, trace):
        """When hook doesn't provide api_call_count, llm_index still works."""
        trace.start_turn(user_message="hi")
        trace.start_llm_call()  # no api_call_count kwarg
        span = trace.end_llm_call(status="completed")
        assert span.metadata["llm_index"] == 1
        assert span.metadata["api_call_count"] == 1  # defaults to llm_index
