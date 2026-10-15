"""The "Ask" page: a chat over the warehouse with visible sources and reasoning steps."""

from __future__ import annotations

import streamlit as st

from ..config import get_settings
from . import data

QUESTIONS_PER_SESSION = 15  # the public app shares one free-tier LLM key
HISTORY_MESSAGES = 6  # earlier turns sent as context (text only)
EXAMPLES = [
    "Which directors have the most films in the Top 250?",
    "What do reviewers say about the pacing of Interstellar?",
    "Which Top 250 movies made the most money relative to their budget?",
    "Do critics and audiences disagree about any Christopher Nolan film?",
]


@st.cache_resource(show_spinner="Loading the search model…")
def _embedder():
    from ..assistant.embedder import Embedder

    return Embedder(get_settings().data_dir / "models" / "bge-small-en-v1.5")


@st.cache_resource
def _client():
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        max_retries=3,
        timeout=60,
    )


def _secrets_hint() -> str | None:
    """Names (never values) of the configured secrets, to diagnose a missing key."""
    try:
        secrets = dict(st.secrets)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return None
    top = sorted(k for k, v in secrets.items() if isinstance(v, str))
    nested = sorted(f"[{k}] {', '.join(v)}" for k, v in secrets.items() if hasattr(v, "keys"))
    hint = f"Secrets found: {', '.join(top) or 'none'}."
    if nested:
        hint += f" Inside sections (not read): {'; '.join(nested)}. Put the key at the top level."
    return hint + " Accepted names: LLM_API_KEY, GROQ_API_KEY, groq."


def _render_answer(answer) -> None:
    st.markdown(answer.text)
    from ..assistant.agent import cited_labels

    cited = cited_labels(answer.text)
    sources = [s for s in answer.sources if s.label in cited]
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                rating = f" · {s.rating:g}/10" if s.rating is not None else ""
                link = f" · [review]({s.url})" if s.url else ""
                st.markdown(f"**[{s.label}] {s.title} ({s.year})** · {s.author}{rating}{link}")
                st.caption(s.text)
    with st.expander("How I got this"):
        for i, step in enumerate(answer.steps, start=1):
            status = f"error: {step.error}" if step.error else f"{step.rows} rows"
            st.markdown(f"**{i}. {step.tool}** · {status} · {step.seconds:.2f}s")
            if step.tool == "run_sql":
                st.code(step.arguments.get("query", ""), language="sql")
            else:
                st.json(step.arguments, expanded=False)
        st.caption(
            f"{answer.seconds:.1f}s · {answer.prompt_tokens:,} prompt + "
            f"{answer.completion_tokens:,} completion tokens · model {get_settings().llm_model}"
        )


def assistant_page() -> None:
    st.title("Ask about the movies")
    st.caption(
        "Answers come from this project's data: SQL over the movie tables and search over "
        "TMDB user reviews. Expand each answer to see its sources and the queries behind it."
    )
    settings = get_settings()
    if not settings.llm_api_key:
        st.info(
            "The assistant needs an LLM key. It works with any OpenAI-compatible endpoint; the "
            "default is Groq's free tier: create a key at console.groq.com and set "
            "`LLM_API_KEY` in `.env` (or in the app's secrets).",
            icon=":material/key:",
        )
        if hint := _secrets_hint():
            st.caption(hint)
        return

    from ..assistant.agent import Agent

    state = st.session_state
    state.setdefault("chat", [])  # list of (question, answer)
    for question, answer in state.chat:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"):
            _render_answer(answer)

    prompt = None
    if not state.chat:
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}"):
                prompt = example
    remaining = QUESTIONS_PER_SESSION - len(state.chat)
    typed = st.chat_input(
        "Ask a question about the movies" if remaining > 0 else "Question limit reached",
        disabled=remaining <= 0,
    )
    prompt = typed or prompt
    if not prompt:
        if remaining <= 0:
            st.caption("This demo allows a limited number of questions per visit.")
        return

    st.chat_message("user").markdown(prompt)
    history = []
    for question, answer in state.chat[-HISTORY_MESSAGES // 2 :]:
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer.text}]  # fmt: skip
    agent = Agent(_client(), settings.llm_model, data.database_path(), _embedder())
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = agent.ask(prompt, history=history)
            except Exception as exc:  # noqa: BLE001 - surface provider errors (rate limits) kindly
                st.error(f"The language model is unavailable right now ({type(exc).__name__}). "
                         "Please try again in a minute.")  # fmt: skip
                return
        _render_answer(answer)
    state.chat.append((prompt, answer))
