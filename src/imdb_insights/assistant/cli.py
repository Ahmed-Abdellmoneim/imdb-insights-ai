"""``uv run python -m imdb_insights.assistant {ask,eval}``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ..config import get_settings
from ..ingest.pipeline import DB_FILENAME

app = typer.Typer(add_completion=False, help="The movie assistant (step 4).")


def build_agent(model: str | None = None):
    from openai import OpenAI

    from .agent import Agent
    from .embedder import Embedder

    settings = get_settings()
    if not settings.llm_api_key:
        typer.echo("Set LLM_API_KEY in .env (Groq's free tier works: console.groq.com).", err=True)
        raise typer.Exit(code=2)
    client = OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        max_retries=5,
        timeout=60,
    )
    embedder = Embedder(settings.data_dir / "models" / "bge-small-en-v1.5")
    db_path = settings.data_dir / "processed" / DB_FILENAME
    return Agent(client, model or settings.llm_model, db_path, embedder), db_path


@app.command()
def ask(question: Annotated[str, typer.Argument(help="A question about the movies.")]) -> None:
    """Answer one question in the terminal, with its sources and steps."""
    agent, _ = build_agent()
    answer = agent.ask(question)
    typer.echo(answer.text + "\n")
    for step in answer.steps:
        detail = step.error or f"{step.rows} rows"
        typer.echo(f"- {step.tool}({json.dumps(step.arguments)}) -> {detail}, {step.seconds}s")
    typer.echo(f"{answer.seconds}s, {answer.prompt_tokens + answer.completion_tokens} tokens")


@app.command("eval")
def evaluate(
    cases: Annotated[Path, typer.Option(help="Evaluation set.")] = Path("evals/assistant.json"),
    model: Annotated[str | None, typer.Option(help="Override LLM_MODEL.")] = None,
) -> None:
    """Run the evaluation set and write reports/assistant_eval.md."""
    from . import evaluate as ev

    agent, db_path = build_agent(model)
    results = ev.run(agent, ev.load_cases(cases), db_path)
    report = ev.summarize(results, agent.model)
    out = Path("reports") / "assistant_eval.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    typer.echo(report)
