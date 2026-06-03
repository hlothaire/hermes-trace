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

from .tracer import (
    TraceGraph,
    TRACE_DIR,
    get_trace,
    get_current_trace,
    remove_trace,
    list_traces,
)


def register(ctx):
    """Register all trace hooks and the /trace slash command."""
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_hook("pre_approval_request", _on_pre_approval_request)
    ctx.register_hook("post_approval_response", _on_post_approval_response)
    ctx.register_hook("on_session_reset", _on_session_reset)

    ctx.register_command(
        "trace",
        handler=_handle_trace_command,
        description="Show the current session's execution trace graph",
    )



def register(ctx):
    """Register session and turn trace hooks."""
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)


def _on_session_start(session_id: str = "", model: str = "", platform: str = "", **kwargs):
    trace = get_trace(session_id)
    trace.model = model
    trace.platform = platform
    trace.started_at = time.time()
    logger.info("Trace started: session=%s model=%s platform=%s", session_id, model, platform)


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
        logger.info(
            "Trace session=%s saved: %s, %s",
            session_id,
            json_path.name,
            mmd_path.name,
        )
    except Exception as exc:
        logger.error("Failed to write trace for session %s: %s", session_id, exc)


def _on_session_finalize(session_id: Optional[str] = None, **kwargs):
    if session_id:
        removed = remove_trace(session_id)
        if removed:
            logger.debug("Trace finalized and removed: %s", session_id)


def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    is_first_turn: bool = False,
    **kwargs,
):
    trace = get_trace(session_id)
    trace.start_turn(user_message=user_message, is_first_turn=is_first_turn)


def _on_post_llm_call(
    session_id: str = "",
    assistant_response: str = "",
    **kwargs,
):
    trace = get_trace(session_id)
    trace.end_turn(assistant_response=assistant_response)


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


def _on_api_request_error(session_id: str = "", error: Any = None, **kwargs):
    trace = get_trace(session_id)
    trace.end_llm_call(
        status="error",
        error=str(error)[:500] if error else "unknown",
    )


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


def _on_api_request_error(session_id: str = "", error: Any = None, **kwargs):
    trace = get_trace(session_id)
    trace.end_llm_call(
        status="error",
        error=str(error)[:500] if error else "unknown",
    )


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
