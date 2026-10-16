"""A tool-calling agent over the warehouse, for any OpenAI-compatible chat endpoint.

The model chooses between two tools: ``run_sql`` for facts and numbers, ``search_reviews``
for what reviewers say. Tool errors (bad SQL, unknown movie) go back to the model as tool
results so it can correct itself. Review passages are numbered [R1], [R2], ... across the
whole turn so the answer can cite them and the UI can list the sources. Every step is
recorded in a trace (tool, arguments, latency, errors, token usage).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from . import search, sql
from .embedder import Embedder

MAX_STEPS = 6
# Zero-width characters some models put inside citations ("[\u200bR2\u200b]"); they would
# break citation matching, copy-paste and search, and carry no meaning in an answer.
_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
# Square brackets, plus the CJK lenticular and full-width brackets GPT-OSS models often use.
_CITATION = re.compile(r"[\[【［]\s*(R\d+)\s*[\]】］]")


def clean_answer(text: str) -> str:
    """Remove invisible characters and normalize citations to the plain ``[R2]`` form."""
    return _CITATION.sub(r"[\1]", _INVISIBLE.sub("", text))


def _is_invalid_tool_call(exc: Exception) -> bool:
    """Provider-side rejection of a malformed tool call (Groq: code "tool_use_failed")."""
    return getattr(exc, "code", None) == "tool_use_failed" or "tool_use_failed" in str(exc)


def cited_labels(text: str) -> set[str]:
    """Source labels cited in an answer, e.g. {"R1", "R4"}."""
    return set(_CITATION.findall(_INVISIBLE.sub("", text)))


PASSAGES_PER_SEARCH = 6

SYSTEM_PROMPT = f"""\
You are the assistant of "IMDb Insights", a project about the best-rated movies. You answer
questions using this project's own data: an IMDb-style Top 250 (ranked with IMDb's weighted
rating) plus thousands of other highly rated candidate films, with IMDb/TMDB/Metacritic/Rotten
Tomatoes ratings, credits, keywords, budgets, awards, and user reviews from TMDB and from
this app's own visitors.

Tools:
- run_sql: facts, numbers, rankings and lists. One read-only DuckDB SELECT per call.
- search_reviews: what reviewers think or say. Returns passages labelled [R1], [R2], ...

{sql.SCHEMA_GUIDE}
Rules:
- Base every factual claim on tool results; never invent numbers, titles or quotes.
- When you use review passages, cite them inline like [R2]. Only cite labels you received.
- If a tool returns an error, fix the call and try again. If the data cannot answer the
  question, say so plainly.
- Politely decline requests unrelated to movies or to this data.
- Be concise. Use a short markdown table for lists of more than three items.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only DuckDB SELECT query over the movie database.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A single SELECT."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Find user-review passages about a topic, optionally for one movie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for in reviews."},
                    "movie": {"type": "string", "description": "Optional movie title."},
                },
                "required": ["query"],
            },
        },
    },
]

# Fingerprint of everything that shapes the model's behaviour apart from the model itself.
# Recorded with every trace and evaluation, so results can be tied to the exact prompt.
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + json.dumps(TOOLS, sort_keys=True)).encode("utf-8")
).hexdigest()[:10]


@dataclass
class Step:
    tool: str
    arguments: dict[str, Any]
    seconds: float
    error: str | None = None
    rows: int | None = None


@dataclass
class Source:
    label: str
    title: str
    year: int | None
    author: str | None
    rating: float | None
    url: str | None
    text: str


@dataclass
class Answer:
    text: str
    steps: list[Step] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    stopped_early: bool = False


class Agent:
    def __init__(
        self, client, model: str, db_path: Path, embedder: Embedder, *, temperature: float = 0
    ) -> None:
        self.client, self.model, self.db_path = client, model, db_path
        self.embedder, self.temperature = embedder, temperature

    # --- tools --------------------------------------------------------------------------

    def _run_sql(self, args: dict, answer: Answer) -> str:
        result = sql.run_query(self.db_path, args["query"])
        answer.steps[-1].rows = len(result.frame)
        return result.to_text()

    def _search_reviews(self, args: dict, answer: Answer) -> str:
        con = search.connect(self.db_path)
        try:
            movie = search.find_movie(con, args["movie"]) if args.get("movie") else None
            if args.get("movie") and movie is None:
                return f"No movie matching {args['movie']!r} in the database."
            hits = search.hybrid_search(
                con,
                self.embedder,
                args["query"],
                k=PASSAGES_PER_SEARCH,
                tmdb_id=movie.tmdb_id if movie else None,
            )
        finally:
            con.close()
        answer.steps[-1].rows = len(hits)
        if hits.empty:
            return "No matching review passages."
        lines = []
        for hit in hits.itertuples():
            label = f"R{len(answer.sources) + 1}"
            answer.sources.append(
                Source(label, hit.title, hit.year, hit.author, hit.rating, hit.url, hit.text)
            )
            rating = f"{hit.rating:g}/10" if pd.notna(hit.rating) else "no rating"
            lines.append(f"[{label}] {hit.title} ({hit.year}), {rating}: {hit.text}")
        return "\n\n".join(lines)

    def _call_tool(self, name: str, raw_args: str, answer: Answer) -> str:
        started = time.monotonic()
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
        answer.steps.append(Step(name, args, 0.0))
        try:
            if name == "run_sql":
                output = self._run_sql(args, answer)
            elif name == "search_reviews":
                output = self._search_reviews(args, answer)
            else:
                raise ValueError(f"Unknown tool {name!r}")
        except (sql.UnsafeQueryError, duckdb.Error, TimeoutError, KeyError, ValueError) as exc:
            answer.steps[-1].error = f"{type(exc).__name__}: {exc}"
            output = f"Error: {answer.steps[-1].error}"
        answer.steps[-1].seconds = round(time.monotonic() - started, 3)
        return output

    # --- loop ---------------------------------------------------------------------------

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        started = time.monotonic()
        answer = Answer(text="")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": question},
        ]
        for _ in range(MAX_STEPS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised unless it is a bad tool call
                if not _is_invalid_tool_call(exc):
                    raise
                # The provider rejected a malformed tool call (e.g. a missing required
                # argument). Record it and let the model try again, like any tool error.
                answer.steps.append(Step("invalid_tool_call", {}, 0.0, error=str(exc)[:300]))
                messages.append(
                    {
                        "role": "user",
                        "content": "Your previous tool call was rejected because its arguments "
                        "did not match the tool's schema. Call the tool again with all required "
                        "arguments.",
                    }
                )
                continue
            if response.usage:
                answer.prompt_tokens += response.usage.prompt_tokens
                answer.completion_tokens += response.usage.completion_tokens
            message = response.choices[0].message
            if not message.tool_calls:
                answer.text = clean_answer(message.content or "")
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in message.tool_calls
                    ],
                }
            )
            for call in message.tool_calls:
                output = self._call_tool(call.function.name, call.function.arguments, answer)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        else:
            answer.stopped_early = True
            answer.text = "I couldn't finish answering within the allowed number of steps."
        answer.seconds = round(time.monotonic() - started, 2)
        return answer
