"""The agent loop, driven by a scripted fake LLM (no network, deterministic)."""

import json
from types import SimpleNamespace

import duckdb
import pytest
from test_assistant_search import PASSAGES, FakeEmbedder

from imdb_insights.assistant import agent
from imdb_insights.assistant.embedder import DIMENSIONS


def _tool_call(i, name, **args):
    return SimpleNamespace(
        id=f"call_{i}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _reply(content=None, calls=None):
    message = SimpleNamespace(content=content, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class ScriptedLLM:
    """Returns the scripted replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return self.replies.pop(0)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "w.duckdb"
    with duckdb.connect(str(path)) as c:
        c.execute(
            "create table movies (tmdb_id bigint, title varchar, original_title varchar, "
            "year int, tmdb_votes int, imdb_rating double, rank int)"
        )
        c.execute(
            "insert into movies values (1, 'Interstellar', 'Interstellar', 2014, 900, 8.7, "
            "20), (2, 'Arrival', 'Arrival', 2016, 500, 7.9, null)"
        )
        c.execute(
            "create table reviews (review_id varchar, author varchar, rating double, url varchar)"
        )
        c.execute(
            f"create table review_chunks (chunk_id varchar, review_id varchar, "
            f"tmdb_id bigint, chunk_no int, text varchar, embedding FLOAT[{DIMENSIONS}])"
        )
        for rid, tmdb_id, text in PASSAGES:
            c.execute("insert into reviews values (?, 'someone', 8, 'https://x/' || ?)", [rid, rid])
            c.execute(
                "insert into review_chunks values (?, ?, ?, 0, ?, ?)",
                [f"{rid}-0", rid, tmdb_id, text, FakeEmbedder._vec(text).tolist()],
            )
        c.execute("install fts; load fts")
        c.execute("pragma create_fts_index('review_chunks', 'chunk_id', 'text')")
    return path


def _agent(llm, db):
    return agent.Agent(llm, "test-model", db, FakeEmbedder())


def test_sql_tool_round_trip(db):
    llm = ScriptedLLM(
        [
            _reply(
                calls=[
                    _tool_call(
                        1, "run_sql", query="select title from movies where rank is not null"
                    )
                ]
            ),
            _reply("The only Top 250 movie here is Interstellar."),
        ]
    )
    answer = _agent(llm, db).ask("Which movies are in the Top 250?")
    assert answer.text.startswith("The only Top 250 movie")
    assert [(s.tool, s.rows, s.error) for s in answer.steps] == [("run_sql", 1, None)]
    tool_message = llm.requests[1][-1]
    assert tool_message["role"] == "tool" and "Interstellar" in tool_message["content"]
    assert (answer.prompt_tokens, answer.completion_tokens) == (200, 20)


def test_sql_error_is_returned_so_the_model_can_fix_it(db):
    llm = ScriptedLLM(
        [
            _reply(calls=[_tool_call(1, "run_sql", query="drop table movies")]),
            _reply(calls=[_tool_call(2, "run_sql", query="select count(*) as n from movies")]),
            _reply("There are 2 movies."),
        ]
    )
    answer = _agent(llm, db).ask("How many movies?")
    assert answer.steps[0].error.startswith("UnsafeQueryError")
    assert llm.requests[1][-1]["content"].startswith("Error: UnsafeQueryError")
    assert answer.steps[1].error is None and answer.text == "There are 2 movies."


def test_review_passages_are_numbered_across_calls(db):
    llm = ScriptedLLM(
        [
            _reply(calls=[_tool_call(1, "search_reviews", query="score", movie="Interstellar")]),
            _reply(calls=[_tool_call(2, "search_reviews", query="score", movie="Arrival")]),
            _reply("Reviewers love the music [R1] and the score [R2]."),
        ]
    )
    answer = _agent(llm, db).ask("What do people say about the music?")
    labels = [s.label for s in answer.sources]
    assert labels == [f"R{i}" for i in range(1, len(labels) + 1)]
    assert {s.title for s in answer.sources} == {"Interstellar", "Arrival"}
    second_result = llm.requests[2][-1]["content"]
    assert second_result.startswith(f"[R{answer.steps[0].rows + 1}]")  # continues numbering


def test_unknown_movie_is_reported_to_the_model(db):
    llm = ScriptedLLM(
        [
            _reply(calls=[_tool_call(1, "search_reviews", query="x", movie="Nope The Movie")]),
            _reply("I don't have that movie."),
        ]
    )
    _agent(llm, db).ask("Reviews of Nope The Movie?")
    assert "No movie matching" in llm.requests[1][-1]["content"]


def test_runaway_loops_stop_after_max_steps(db):
    calls = [
        _reply(calls=[_tool_call(i, "run_sql", query="select 1")]) for i in range(agent.MAX_STEPS)
    ]
    answer = _agent(ScriptedLLM(calls), db).ask("Loop forever")
    assert answer.stopped_early and len(answer.steps) == agent.MAX_STEPS


def test_history_and_system_prompt_are_sent(db):
    llm = ScriptedLLM([_reply("Hi!")])
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    _agent(llm, db).ask("Hello", history=history)
    sent = llm.requests[0]
    assert sent[0]["role"] == "system" and "run_sql" in sent[0]["content"]
    assert [m["content"] for m in sent[1:]] == ["earlier", "ok", "Hello"]


def test_citations_with_invisible_characters_are_normalized():
    # GPT-OSS on Groq writes citations with zero-width spaces inside the brackets.
    raw = "Iconic [\u200bR2\u200b] and magnetic [ R4 ]. Not a citation: [Rx]."
    assert agent.cited_labels(raw) == {"R2", "R4"}
    assert agent.clean_answer(raw) == "Iconic [R2] and magnetic [R4]. Not a citation: [Rx]."
    # ...and sometimes CJK lenticular or full-width brackets.
    assert agent.cited_labels("Stunning\u3010R1\u3011\u3010R3\u3011 and \uff3bR5\uff3d") == {
        "R1",
        "R3",
        "R5",
    }
    assert agent.clean_answer("Stunning\u3010R1\u3011") == "Stunning[R1]"


class ToolUseFailed(Exception):
    code = "tool_use_failed"


class FlakyLLM(ScriptedLLM):
    """Rejects the first request the way Groq rejects a malformed tool call."""

    def _create(self, **kwargs):
        if not self.requests:
            self.requests.append(kwargs["messages"])
            raise ToolUseFailed("parameters for tool run_sql did not match schema")
        return super()._create(**kwargs)


def test_rejected_tool_calls_are_retried(db):
    llm = FlakyLLM([_reply("There are 2 movies.")])
    answer = _agent(llm, db).ask("How many movies?")
    assert answer.text == "There are 2 movies."
    assert answer.steps[0].tool == "invalid_tool_call" and answer.steps[0].error
    assert "rejected" in llm.requests[-1][-1]["content"]


def test_other_provider_errors_still_raise(db):
    class Broken(ScriptedLLM):
        def _create(self, **kwargs):
            raise RuntimeError("service unavailable")

    with pytest.raises(RuntimeError):
        _agent(Broken([]), db).ask("Anything")
