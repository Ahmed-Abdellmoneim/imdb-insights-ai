"""A tool-calling agent over the warehouse, for any OpenAI-compatible chat endpoint.

The model chooses between two tools: ``run_sql`` for facts and numbers, ``search_reviews``
for what reviewers say. Tool errors (bad SQL, unknown movie) go back to the model as tool
results so it can correct itself. Review passages are numbered [R1], [R2], ... across the
whole turn so the answer can cite them and the UI can list the sources. Every step is
recorded in a trace (tool, arguments, latency, errors, token usage).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from . import search, sql
from .embedder import Embedder

MAX_STEPS = 6
PASSAGES_PER_SEARCH = 6

SYSTEM_PROMPT = f"""\
You are the assistant of "IMDb Insights", a project about the best-rated movies. You answer
questions using this project's own data: an IMDb-style Top 250 (ranked with IMDb's weighted
rating) plus thousands of other highly rated candidate films, with IMDb/TMDB/Metacritic/Rotten
Tomatoes ratings, credits, keywords, budgets, awards, and user reviews from TMDB.

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
            )
            if response.usage:
                answer.prompt_tokens += response.usage.prompt_tokens
                answer.completion_tokens += response.usage.completion_tokens
            message = response.choices[0].message
            if not message.tool_calls:
                answer.text = message.content or ""
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
