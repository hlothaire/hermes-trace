"""CLI handler for `hermes trace` — query traces outside a session.

Subcommands:
  list              List saved trace files
  view <session_id> Show text tree for a trace
  stats <session_id> Show aggregate statistics
  clean --keep N    Rotate old traces, keeping the N most recent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .tracer import TraceGraph, TRACE_DIR


def _list_traces() -> None:
    """Print a table of saved traces."""
    if not TRACE_DIR.exists():
        print("No traces directory found.")
        return

    json_files = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        print("No saved traces.")
        return

    print(f"{'SESSION ID':<30} {'TURNS':>6} {'TOKENS IN':>10} {'TOKENS OUT':>10} {'DURATION':>10} {'DATE':>20}")
    print("-" * 90)

    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            print(f"{f.stem:<30} {'(corrupt)':>50}")
            continue

        sid = data.get("session_id", f.stem)[:28]
        turns = len(data.get("turns", []))
        total_in = 0
        total_out = 0
        for t in data.get("turns", []):
            for s in t.get("spans", []):
                if s.get("kind") == "llm_call":
                    u = s.get("metadata", {}).get("usage", {})
                    total_in += u.get("input_tokens", 0) or 0
                    total_out += u.get("output_tokens", 0) or 0

        dur = data.get("duration_s", 0)
        dur_str = f"{dur:.0f}s" if dur else "?"

        import datetime
        mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        date_str = mtime.strftime("%Y-%m-%d %H:%M")

        print(f"{sid:<30} {turns:>6} {total_in:>10,} {total_out:>10,} {dur_str:>10} {date_str:>20}")

    print(f"\n{len(json_files)} trace(s) in {TRACE_DIR}")


def _view_trace(session_id: str) -> None:
    """Show the text tree for a saved trace."""
    json_path = TRACE_DIR / f"{session_id}.json"
    if not json_path.exists():
        # Try partial match
        matches = sorted(TRACE_DIR.glob(f"{session_id}*.json"))
        if not matches:
            print(f"No trace found for session '{session_id}'.")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Ambiguous session ID. Candidates: {', '.join(m.stem for m in matches)}")
            sys.exit(1)
        json_path = matches[0]

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        trace = TraceGraph.from_dict(data)
        print(trace.to_text_tree())
    except Exception as exc:
        print(f"Failed to load trace: {exc}")
        sys.exit(1)


def _stats_trace(session_id: str) -> None:
    """Show aggregate statistics for a saved trace."""
    json_path = TRACE_DIR / f"{session_id}.json"
    if not json_path.exists():
        matches = sorted(TRACE_DIR.glob(f"{session_id}*.json"))
        if not matches:
            print(f"No trace found for session '{session_id}'.")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Ambiguous session ID. Candidates: {', '.join(m.stem for m in matches)}")
            sys.exit(1)
        json_path = matches[0]

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        trace = TraceGraph.from_dict(data)
        stats = trace.compute_stats()

        print(f"Session:  {data.get('session_id', session_id)}")
        print(f"Model:    {data.get('model', '?')}")
        print(f"Platform: {data.get('platform', '?')}")
        print(f"Duration: {data.get('duration_s', 0):.0f}s")
        print()
        print(f"Turns:      {stats['turns']}")
        print(f"LLM calls:  {stats['llm_calls']}")
        print(f"Tool calls: {stats['tool_calls']}")
        if stats["errors"]:
            print(f"Errors:     {stats['errors']}")
        print(f"Tokens:     {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out")

        slow = stats.get("slowest_llm")
        if slow:
            print(f"Slowest LLM:   #{slow['api_call_count']} ({slow['duration_ms'] / 1000:.1f}s)")
        slow = stats.get("slowest_tool")
        if slow:
            print(f"Slowest tool:  {slow['name']} ({slow['duration_ms'] / 1000:.1f}s)")

    except Exception as exc:
        print(f"Failed to load trace: {exc}")
        sys.exit(1)


def _clean_traces(keep: int) -> None:
    """Delete old traces, keeping the N most recent JSON files.

    Also cleans up orphaned .mmd files without a matching .json.
    """
    if not TRACE_DIR.exists():
        print("No traces directory.")
        return

    json_files = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    deleted = 0
    for f in json_files[keep:]:
        # Delete JSON + matching Mermaid file
        f.unlink()
        deleted += 1
        mmd = f.with_suffix(".mmd")
        if mmd.exists():
            mmd.unlink()

    # Clean orphaned .mmd files
    json_stems = {f.stem for f in json_files[:keep]}
    for mmd in TRACE_DIR.glob("*.mmd"):
        if mmd.stem not in json_stems:
            mmd.unlink()
            deleted += 1

    kept = min(len(json_files), keep)
    print(f"Kept {kept} trace(s), deleted {deleted} file(s).")


# ---- argparse integration --------------------------------------------------


def _setup_argparse(subparser: argparse.ArgumentParser) -> None:
    """Build the argparse tree for `hermes trace`."""
    subs = subparser.add_subparsers(dest="trace_command", help="Subcommand")

    # list
    subs.add_parser("list", help="List saved traces")

    # view <session_id>
    view_p = subs.add_parser("view", help="Show text tree for a trace")
    view_p.add_argument("session_id", help="Session ID (can be a prefix)")

    # stats <session_id>
    stats_p = subs.add_parser("stats", help="Show aggregate statistics")
    stats_p.add_argument("session_id", help="Session ID (can be a prefix)")

    # clean --keep N
    clean_p = subs.add_parser("clean", help="Rotate old traces")
    clean_p.add_argument("--keep", type=int, default=20, help="Number of recent traces to keep (default: 20)")

    subparser.set_defaults(func=_handle_command)


def _handle_command(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate subcommand handler."""
    cmd = getattr(args, "trace_command", None)

    if cmd == "list":
        _list_traces()
    elif cmd == "view":
        _view_trace(args.session_id)
    elif cmd == "stats":
        _stats_trace(args.session_id)
    elif cmd == "clean":
        _clean_traces(args.keep)
    else:
        # No subcommand given — show help
        print("Usage: hermes trace <list|view <id>|stats <id>|clean>")
        print()
        print("Subcommands:")
        print("  list              List saved traces")
        print("  view <session_id> Show text tree for a trace")
        print("  stats <session_id> Show aggregate statistics")
        print("  clean --keep N    Rotate old traces (default: keep 20)")
        sys.exit(1)
