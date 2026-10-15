"""Traces of assistant turns, and a report over them.

Each answered question becomes one JSON line: when, which model and prompt version, the
question and answer, every tool step (arguments, rows, error, latency), citations, token
usage and total latency. JSON Lines keep writes append-only and cheap, and DuckDB reads them
directly for the report. Review passages are stored by label and title only, not their text.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .agent import PROMPT_VERSION, Answer, cited_labels

_lock = threading.Lock()  # Streamlit serves sessions from several threads


def trace_record(
    question: str, answer: Answer, *, model: str, source: str, session: str | None = None
) -> dict[str, Any]:
    cited = cited_labels(answer.text)
    return {
        "trace_id": uuid.uuid4().hex,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,  # "app", "cli" or "eval"
        "session": session,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "question": question,
        "answer": answer.text,
        "seconds": answer.seconds,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "stopped_early": answer.stopped_early,
        "steps": [asdict(step) for step in answer.steps],
        "tool_errors": sum(1 for step in answer.steps if step.error),
        "sources_received": len(answer.sources),
        "sources_cited": sorted(cited),
        "cited_titles": sorted({s.title for s in answer.sources if s.label in cited}),
    }


def write_trace(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def report(path: Path) -> str:
    """Latency, tokens, errors and tool use per model and prompt version, as markdown."""
    if not path.exists() or path.stat().st_size == 0:
        return "No traces yet."
    con = duckdb.connect()
    # A view cannot take query parameters; the path is our own setting, quoted as a literal.
    literal = "'" + path.as_posix().replace("'", "''") + "'"
    con.execute(
        f"create view traces as select * from read_json_auto({literal}, format='newline_delimited')"
    )
    overview = con.sql(
        """
        select model, prompt_version, count(*) as questions,
               round(quantile_cont(seconds, 0.5), 1) as p50_s,
               round(quantile_cont(seconds, 0.95), 1) as p95_s,
               round(avg(prompt_tokens + completion_tokens)) as avg_tokens,
               round(100 * avg((tool_errors > 0)::int), 1) as pct_with_tool_error,
               round(100 * avg(stopped_early::int), 1) as pct_stopped_early,
               round(100 * avg((len(steps) = 0)::int), 1) as pct_no_tools,
               min(time) as first, max(time) as last
        from traces group by all order by last desc
        """
    ).df()
    tools = con.sql(
        """
        select s.tool, count(*) as calls,
               round(quantile_cont(s.seconds, 0.5) * 1000) as p50_ms,
               round(100 * avg((s.error is not null)::int), 1) as pct_error
        from traces, unnest(steps) as u(s) group by s.tool order by calls desc
        """
    ).df()
    errors = con.sql(
        """
        select s.tool, left(regexp_replace(s.error, '\\s+', ' ', 'g'), 120) as error,
               count(*) as n
        from traces, unnest(steps) as u(s) where s.error is not null
        group by all order by n desc limit 10
        """
    ).df()
    parts = ["# Assistant traces", "", "## By model and prompt version", "",
             _table(overview), "", "## Tools", "", _table(tools)]  # fmt: skip
    if not errors.empty:
        parts += ["", "## Most common tool errors", "", _table(errors)]
    return "\n".join(parts) + "\n"


def _table(df) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])
