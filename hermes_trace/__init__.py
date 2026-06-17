"""hermes-trace plugin — build trace graphs of agent execution.

A Hermes plugin that captures the agent's execution as a directed graph:
  Session → Turns → (LLM calls → Tool calls, Subagents)

Provides:
  - Automatic tracing of all agent activity via lifecycle hooks
  - /trace slash command to view, export, and manage traces
  - JSON and Mermaid diagram output at session end
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Map child_session_id → parent_session_id for cross-trace navigation.
# Populated by subagent_stop, consumed by on_session_end.
_pending_parent_links: dict[str, str] = {}

from .tracer import (
    TraceGraph,
    TRACE_DIR,
    get_trace,
    get_current_trace,
    remove_trace,
    list_traces,
)


# ---- Error-resilient hook wrapper ------------------------------------------


def _trace_hook(name: str):
    """Decorator that wraps a hook callback in try/except.

    On exception, logs the error and returns None — the agent
    continues normally.  One broken callback never orphans spans
    or crashes the agent.
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception("Trace hook '%s' crashed — continuing", name)
                return None
        return wrapper
    return decorator


def register(ctx):
    """Register all trace hooks and the /trace slash command."""
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_hook("pre_approval_request", _on_pre_approval_request)
    ctx.register_hook("post_approval_response", _on_post_approval_response)
    ctx.register_hook("on_session_reset", _on_session_reset)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)

    ctx.register_command(
        "trace",
        handler=_handle_trace_command,
        description="Show the current session's execution trace graph",
    )

    from .cli import _setup_argparse, _handle_command
    ctx.register_cli_command(
        name="trace",
        help="Query and manage trace graphs",
        setup_fn=_setup_argparse,
        handler_fn=_handle_command,
    )


# ---- Hook callbacks -------------------------------------------------------


def _patch_parent_trace(json_path: Path, parent_session_id: str) -> None:
    """Patch a trace JSON file to include a ``parent_trace`` reference.

    Used for subagent child → parent navigation.  Never raises.
    """
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "parent_trace" not in data:
            data["parent_trace"] = parent_session_id
            json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            logger.debug("Patched parent_trace=%s into %s", parent_session_id, json_path.name)
    except Exception:
        logger.debug("Failed to read/patch %s", json_path.name)


@_trace_hook("on_session_start")
def _on_session_start(session_id: str = "", model: str = "", platform: str = "", **kwargs):
    trace = get_trace(session_id)
    trace.model = model
    trace.platform = platform
    trace.started_at = time.time()
    logger.info("Trace started: session=%s model=%s platform=%s", session_id, model, platform)


@_trace_hook("on_session_end")
def _on_session_end(
    session_id: str = "",
    completed: bool = False,
    interrupted: bool = False,
    **kwargs,
):
    trace = get_trace(session_id)
    trace.finalize(completed=completed, interrupted=interrupted)

    # Write trace files
    try:
        json_path = trace.write_json()
        mmd_path = trace.write_mermaid()

        # If this session is a subagent child, patch parent_trace into JSON
        parent_sid = _pending_parent_links.pop(session_id, None)
        if parent_sid:
            try:
                _patch_parent_trace(json_path, parent_sid)
            except Exception:
                logger.debug("Failed to patch parent_trace for child %s", session_id)

        logger.info(
            "Trace session=%s saved: %s, %s",
            session_id,
            json_path.name,
            mmd_path.name,
        )
    except Exception as exc:
        logger.error("Failed to write trace for session %s: %s", session_id, exc)


@_trace_hook("on_session_finalize")
def _on_session_finalize(session_id: Optional[str] = None, **kwargs):
    if session_id:
        removed = remove_trace(session_id)
        if removed:
            logger.debug("Trace finalized and removed: %s", session_id)


@_trace_hook("pre_llm_call")
def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    is_first_turn: bool = False,
    **kwargs,
):
    trace = get_trace(session_id)
    # Backfill started_at if session_start hook fired before agent initialisation
    trace.ensure_started()
    trace.start_turn(user_message=user_message, is_first_turn=is_first_turn)


@_trace_hook("post_llm_call")
def _on_post_llm_call(
    session_id: str = "",
    assistant_response: str = "",
    **kwargs,
):
    trace = get_trace(session_id)
    trace.end_turn(assistant_response=assistant_response)


@_trace_hook("pre_api_request")
def _on_pre_api_request(
    session_id: str = "",
    api_call_count: int = 0,
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    message_count: int = 0,
    tool_count: int = 0,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    max_tokens: int = 0,
    **kwargs,
):
    trace = get_trace(session_id)
    # Backfill session-level metadata if session_start hook fired too early
    if not trace.model and model:
        trace.model = model
        logger.debug("Trace: backfilled model=%s for session %s", model, session_id)
    if not trace.platform and provider:
        trace.platform = provider
    trace.start_llm_call(
        api_call_count=api_call_count,
        model=model,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        message_count=message_count,
        tool_count=tool_count,
        approx_input_tokens=approx_input_tokens,
        request_char_count=request_char_count,
        max_tokens=max_tokens,
    )


@_trace_hook("post_api_request")
def _on_post_api_request(
    session_id: str = "",
    usage: Optional[dict] = None,
    finish_reason: str = "",
    response_model: str = "",
    api_duration: float = 0,
    **kwargs,
):
    trace = get_trace(session_id)
    trace.end_llm_call(
        status="completed",
        usage=usage or {},
        finish_reason=finish_reason,
        response_model=response_model,
        api_duration=api_duration,
    )


@_trace_hook("api_request_error")
def _on_api_request_error(session_id: str = "", error: Any = None, **kwargs):
    trace = get_trace(session_id)
    trace.end_llm_call(
        status="error",
        error=str(error)[:500] if error else "unknown",
    )


@_trace_hook("pre_tool_call")
def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[dict] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **kwargs,
):
    trace = get_trace(session_id or task_id)
    trace.start_tool_call(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        args=args or {},
    )


@_trace_hook("post_tool_call")
def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[dict] = None,
    result: str = "",
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    status: str = "",
    error_type: str = "",
    error_message: str = "",
    **kwargs,
):
    trace = get_trace(session_id or task_id)
    # Use the hook-provided status if available; fall back to result inspection
    if not status and result:
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict) and "error" in parsed:
                status = "error"
        except (json.JSONDecodeError, TypeError):
            pass
    if not status:
        status = "completed"

    trace.end_tool_call(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        status=status,
        result_preview=result[:500] if result else "",
        duration_ms=duration_ms,
    )


@_trace_hook("subagent_start")
def _on_subagent_start(
    session_id: str = "",
    **kwargs,
):
    child_session_id = kwargs.get("child_session_id", kwargs.get("child_task_id", "?"))
    goal = kwargs.get("child_goal", kwargs.get("task_goal", ""))
    trace = get_trace(session_id)
    trace.start_subagent(
        child_session_id=str(child_session_id),
        goal=str(goal),
    )


@_trace_hook("subagent_stop")
def _on_subagent_stop(
    parent_session_id: str = "",
    child_status: str = "completed",
    child_summary: str = "",
    duration_ms: int = 0,
    **kwargs,
):
    child_session_id = kwargs.get("child_session_id", kwargs.get("child_task_id", "?"))
    child_sid = str(child_session_id)
    parent_sid = str(parent_session_id)

    trace = get_trace(parent_sid)
    trace.end_subagent(
        child_session_id=child_sid,
        status=child_status,
        summary=child_summary or "",
        duration_ms=duration_ms,
    )

    # Store parent link for cross-trace navigation
    if child_sid and parent_sid:
        _pending_parent_links[child_sid] = parent_sid

        # If the child's trace JSON already exists, patch it now
        child_json = TRACE_DIR / f"{child_sid}.json"
        if child_json.exists():
            try:
                _patch_parent_trace(child_json, parent_sid)
            except Exception:
                logger.debug("Failed to patch parent_trace into %s", child_sid)


@_trace_hook("pre_approval_request")
def _on_pre_approval_request(
    command: str = "",
    description: str = "",
    pattern_key: str = "",
    session_key: str = "",
    surface: str = "",
    **kwargs,
):
    """Capture approval requests as trace events."""
    if not session_key:
        return
    trace = get_trace(session_key)
    # Attach to current turn, or most recent turn if none is active
    turn = trace._current_turn or (trace.turns[-1] if trace.turns else None)
    if turn:
        turn.metadata.setdefault("approvals", []).append({
            "event": "requested",
            "command": command,
            "description": description,
            "pattern_key": pattern_key,
            "surface": surface,
        })


@_trace_hook("post_approval_response")
def _on_post_approval_response(
    command: str = "",
    description: str = "",
    pattern_key: str = "",
    session_key: str = "",
    surface: str = "",
    choice: str = "",
    **kwargs,
):
    """Capture approval responses as trace events."""
    if not session_key:
        return
    trace = get_trace(session_key)
    turn = trace._current_turn or (trace.turns[-1] if trace.turns else None)
    if turn:
        turn.metadata.setdefault("approvals", []).append({
            "event": "responded",
            "command": command,
            "choice": choice,
            "surface": surface,
        })


# ---- Transform hooks (observer only — never modify the payload) ----------


@_trace_hook("transform_tool_result")
def _on_transform_tool_result(
    tool_name: str = "",
    args: Optional[dict] = None,
    result: str = "",
    task_id: str = "",
    session_id: str = "",
    **kwargs,
):
    """Capture tool result as seen by the model after all transforms.

    Fires after post_tool_call but before the result is handed back to
    the LLM.  We annotate the most recently ended tool span in the
    current turn so the trace reflects what the model actually received.
    """
    # Return None → never modify the result (observer only)
    sid = session_id or task_id
    if not sid:
        return None
    trace = get_trace(sid)
    turn = trace._current_turn or (trace.turns[-1] if trace.turns else None)
    if turn and turn.spans:
        # Walk backwards to find the most recent tool_call span
        for span in reversed(turn.spans):
            if span.kind == "tool_call" and span.name == tool_name:
                span.metadata["transformed_result"] = result[:500] if result else ""
                break
    return None


@_trace_hook("transform_llm_output")
def _on_transform_llm_output(
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: Optional[list] = None,
    model: str = "",
    platform: str = "",
    **kwargs,
):
    """Capture the final assistant response after all transforms.

    Fires after post_llm_call but before the response is delivered to
    the user.  We store the transformed version on the current turn so
    the trace shows what the user actually received.
    """
    if not session_id:
        return None
    trace = get_trace(session_id)
    turn = trace._current_turn or (trace.turns[-1] if trace.turns else None)
    if turn and assistant_response:
        turn.metadata["transformed_response"] = assistant_response[:500]
    return None  # never modify the response


@_trace_hook("pre_gateway_dispatch")
def _on_pre_gateway_dispatch(
    event: Any = None,
    gateway: Any = None,
    session_store: Any = None,
    **kwargs,
):
    """Capture gateway message dispatch events.

    Fires once per incoming MessageEvent in the gateway, before auth /
    pairing / agent dispatch.  Since no session exists yet we record a
    lightweight event that can be correlated once on_session_start fires.
    """
    if event is None:
        return None
    src = getattr(event, "source", "?")
    text = getattr(event, "text", "")[:200] if hasattr(event, "text") else ""
    # Log only — full trace correlation happens when the session starts
    logger.debug(
        "Trace: gateway dispatch source=%s text=%s",
        src,
        text,
    )
    return None


@_trace_hook("on_session_reset")
def _on_session_reset(session_id: str = "", platform: str = "", **kwargs):
    """Record session reset events."""
    if not session_id:
        return
    trace = get_trace(session_id)
    trace.metadata.setdefault("lifecycle", []).append({
        "event": "reset",
        "platform": platform,
    })


# ---- Slash command handler ------------------------------------------------


def _handle_trace_command(raw_args: str) -> str:
    """Handler for /trace — show the current trace graph as a text tree."""
    args = raw_args.strip().split()
    action = args[0] if args else "view"

    if action == "list":
        traces = list_traces()
        if not traces:
            if TRACE_DIR.exists():
                json_files = sorted(TRACE_DIR.glob("*.json"), reverse=True)
                if json_files:
                    lines = ["Past traces on disk:"]
                    for f in json_files[:10]:
                        lines.append(f"  - {f.name}")
                    return "\n".join(lines)
            return "No active or saved traces found."
        return "Active traces:\n" + "\n".join(f"  - {t}" for t in traces)

    if action == "view":
        session_id = args[1] if len(args) > 1 else None
        trace = get_current_trace(session_id)
        if trace is None:
            if session_id:
                json_path = TRACE_DIR / f"{session_id}.json"
                if json_path.exists():
                    return f"Trace exists on disk: {json_path}"
                return f"No trace found for session {session_id}."
            if TRACE_DIR.exists():
                json_files = sorted(TRACE_DIR.glob("*.json"), reverse=True)
                if json_files:
                    sid = json_files[0].stem
                    return (
                        "No active trace in memory.\n"
                        f"Most recent saved trace: {sid}\n"
                        f"Use /trace load {sid} to view it."
                    )
            return "No active trace. Start a conversation first!"
        return trace.to_text_tree()

    if action == "load":
        if len(args) < 2:
            return "Usage: /trace load <session_id>"
        session_id = args[1]
        json_path = TRACE_DIR / f"{session_id}.json"
        if not json_path.exists():
            return f"No saved trace found at {json_path}"
        try:
            data = json.loads(json_path.read_text())
            trace = TraceGraph.from_dict(data)
            return trace.to_text_tree()
        except Exception as exc:
            return f"Failed to load trace: {exc}"

    if action in ("export", "save"):
        trace = get_current_trace()
        if trace is None:
            return "No active trace to export."
        try:
            json_path = trace.write_json()
            mmd_path = trace.write_mermaid()
            return f"Trace exported:\n  JSON: {json_path}\n  Mermaid: {mmd_path}"
        except Exception as exc:
            return f"Export failed: {exc}"

    if action == "mermaid":
        trace = get_current_trace()
        if trace is None:
            return "No active trace."
        return trace.to_mermaid()

    if action == "clear":
        trace = get_current_trace()
        if trace is not None and trace.session_id:
            remove_trace(trace.session_id)
            return f"Cleared active trace for session {trace.session_id}."
        return "No active trace to clear."

    if action == "gantt":
        trace = get_current_trace()
        if trace is None:
            return "No active trace."
        return trace.to_gantt()

    if action == "html":
        trace = get_current_trace()
        if trace is None:
            return "No active trace."
        try:
            path = trace.write_html()
            return f"HTML trace written to {path}"
        except Exception as exc:
            return f"HTML export failed: {exc}"

    return (
        "Usage: /trace [view|list|load <id>|export|mermaid|gantt|html|clear]\n"
        "  view     - Show current trace as text tree (default)\n"
        "  list     - List active/saved traces\n"
        "  load <id> - Load and display a saved trace\n"
        "  export   - Save current trace to ~/.hermes/traces/\n"
        "  mermaid  - Show trace as Mermaid flowchart\n"
        "  gantt    - Show ASCII Gantt timeline\n"
        "  html     - Export interactive HTML timeline\n"
        "  clear    - Remove current trace from memory"
    )
