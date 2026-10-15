import json
import threading

from imdb_insights.assistant import tracing
from imdb_insights.assistant.agent import PROMPT_VERSION, Answer, Source, Step


def _answer(seconds=1.0, error=None, text="Fine [R1]."):
    return Answer(
        text=text,
        steps=[Step("search_reviews", {"query": "x"}, 0.2, error=error, rows=3)],
        sources=[
            Source("R1", "Interstellar", 2014, "a", 8.0, None, "secret passage text"),
            Source("R2", "Arrival", 2016, "b", 7.0, None, "other text"),
        ],
        prompt_tokens=100,
        completion_tokens=20,
        seconds=seconds,
    )


def test_record_has_versions_steps_and_citations_but_not_passage_text():
    record = tracing.trace_record("q?", _answer(), model="m", source="cli")
    assert record["prompt_version"] == PROMPT_VERSION and len(PROMPT_VERSION) == 10
    assert record["steps"][0]["tool"] == "search_reviews"
    assert record["sources_cited"] == ["R1"] and record["cited_titles"] == ["Interstellar"]
    assert "secret passage text" not in json.dumps(record)


def test_concurrent_writes_produce_valid_json_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    record = tracing.trace_record("q", _answer(), model="m", source="app")

    def write_many():
        for _ in range(50):
            tracing.write_trace(path, record)

    threads = [threading.Thread(target=write_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200 and all(json.loads(line)["question"] == "q" for line in lines)


def test_report_summarizes_latency_errors_and_tools(tmp_path):
    path = tmp_path / "t.jsonl"
    for seconds in (1, 2, 3, 4):
        tracing.write_trace(
            path, tracing.trace_record("q", _answer(seconds), model="m", source="eval")
        )
    failing = _answer(9, error="UnsafeQueryError: nope")
    tracing.write_trace(path, tracing.trace_record("q", failing, model="m", source="eval"))
    text = tracing.report(path)
    assert "| m | " + PROMPT_VERSION + " | 5 | 3.0 |" in text  # 5 questions, median 3.0 s
    assert "| search_reviews | 5 |" in text
    assert "UnsafeQueryError: nope" in text and "| 20.0 |" in text  # 1 of 5 had an error


def test_report_without_traces(tmp_path):
    assert tracing.report(tmp_path / "missing.jsonl") == "No traces yet."
