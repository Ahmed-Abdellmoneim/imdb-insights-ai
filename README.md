# IMDb Insights AI

An end-to-end data and AI project around the best-rated movies of all time: a resumable,
API-based ingestion pipeline, an analytical DuckDB warehouse, an IMDb-style Top 250 we compute
ourselves, and (in progress) NLP over reviews and an AI assistant.

> **v2** of [IMDb-Top-250-Scraper-Review](https://github.com/Ahmed-Abdellmoneim/IMDb-Top-250-Scraper-Review)
> (2024). IMDb has since blocked automated browsers and put reviews behind sign-in, so v2
> collects data through official APIs instead of scraping — see [Why APIs](#why-apis-not-scraping).

## What it does today

```
TMDB discover ──► candidates (rating ≥ 6.9 with ≥ 250 votes, plus ≥ 7.0 with 50–249 votes)
      │
      ├─► TMDB /movie/{id}  details · cast & crew · keywords · reviews · IMDb id
      └─► OMDb ?i={imdb_id} IMDb rating & votes · Metascore · Rotten Tomatoes · awards
                 │
                 ▼
     weighted-rating ranking ──► quality checks ──► DuckDB + Parquet + manifest.json
```

- **Async API clients** with bounded concurrency, request pacing, retries with exponential
  backoff (honouring `Retry-After`), and an on-disk response cache.
- **Resumable under quotas.** OMDb's free tier allows 1,000 requests/day. When it is exhausted
  the pipeline stops asking, writes what it has, marks the rest `pending`, and exits with code
  `3`; the next run only fetches what is missing.
- **Validated records.** Every API response is parsed into Pydantic models; API quirks
  (`"N/A"`, `"$28,767,189"`, `0` meaning unknown, errors inside HTTP 200) are normalized.
- **Our own Top 250**, using the Bayesian weighted rating IMDb published for its chart:
  `WR = v/(v+m)·R + m/(v+m)·C` with `m = 25,000` votes and `C = 7.0`.
- **Self-checks** in `manifest.json`: whether the run is complete, and whether the candidate
  pool was wide enough that no Top 250 movie could have been missed.

### Data model (DuckDB: `data/processed/imdb_insights.duckdb`)

| table | grain | highlights |
|---|---|---|
| `movies` | one row per candidate | TMDB + IMDb ratings, votes, Metascore, RT, budget, revenue, awards, `weighted_rating`, `rank` |
| `movie_genres` | movie × genre | |
| `movie_keywords` | movie × keyword | |
| `people` | one row per person | name, gender, profile image |
| `credits` | movie × person × role | cast (character, billing) and crew (department, job) |
| `reviews` | one row per review | text, reviewer rating, date (TMDB) |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and free API keys from
[TMDB](https://www.themoviedb.org/settings/api) and [OMDb](https://www.omdbapi.com/apikey.aspx).

```bash
uv sync
cp .env.example .env        # then fill in TMDB_API_KEY (read access token) and OMDB_API_KEY
uv run python -m imdb_insights.ingest
uv run python -m streamlit run streamlit_app.py
uv run python -m pytest
```

The app has four pages: the **Top 250** (filterable, with posters), a **Movie** page (ratings
from four sources, crew, cast, keywords, reviews; shareable via `?id=<tmdb id>`),
**Insights** (decades, genres, directors, actors, IMDb vs TMDB, critics vs audiences, budget vs
box office; every chart has a table view), and **About the data** (method and latest run).
The pipeline can refresh the data while the app is running: it builds the new database in a
temporary file and swaps it in with one rename.

Options: `--min-rating`, `--min-votes`, `--data-dir` (see `--help`). A first full run needs
about two days of the OMDb free quota; re-running is safe and cheap because responses are cached.

> **Windows note:** if an Application Control policy blocks the `.exe` launchers that uv
> creates (`os error 4551`), run tools as modules (`uv run python -m pytest`), as above.

## Deployment

```
GitHub Actions (daily) ──► pull API cache ◄── private Hugging Face dataset ──► Streamlit app
        │                                            ▲                       (syncs hourly,
        └─► ingest (only new / expired data) ─► push warehouse + cache       read-only token)
```

- **Private data, public code.** TMDB data may not be redistributed in bulk, so the warehouse
  lives in a private Hugging Face dataset; `publish` refuses to write to a public one and
  squashes history on every push so storage stays flat.
- **Incremental.** Cached responses expire after ~7 days (TMDB) and ~21 days (OMDb), spread
  per key so refreshes are distributed across days and stay inside OMDb's daily quota.
  Movies with no rating at all are fetched first; if a refresh fails, the last known value
  is kept and counted in the manifest.
- **Zero-downtime refresh.** The app polls for a new warehouse at most hourly and swaps it in
  atomically; a failed download keeps the current data.

Configuration: repository secrets `TMDB_API_KEY`, `OMDB_API_KEY`, `HF_TOKEN` (write) and the
variable `HF_DATASET_REPO`; the Streamlit app needs `HF_TOKEN` (read-only) and `HF_DATASET_REPO`.

## Review rating models

Predict the 1-10 star rating a reviewer gave, from the review text. Full results with 95%
bootstrap intervals: [`reports/review_models.md`](reports/review_models.md).

**Data.** Stanford Large Movie Review Dataset (25k train / 25k test, official movie-disjoint
split) plus TMDB reviews from the warehouse as an out-of-domain test set. Validation is a
movie-grouped slice of train. Measured and handled: 123 test reviews duplicated verbatim in
train and one movie in both halves (flagged and excluded from the clean test set), and
reviews that state their own "x/10" score (reported as a separate slice).

| Clean Stanford test (24,873) | MAE ↓ | within ±1 ↑ | pos/neg accuracy ↑ | TMDB Spearman ↑ |
|---|---|---|---|---|
| predict the mean | 3.29 | 10.5% | 50.0% | – |
| TF-IDF (1-2 grams) + ridge | 1.61 | 53.7% | 89.4% | 0.50 |
| bge-small embeddings + ridge | 1.25 | 68.6% | 92.9% | 0.58 |
| **DistilBERT fine-tuned** (regression head, head+tail truncation) | **1.03** | **80.6%** | **93.6%** | **0.69** |

**Specialist vs. general LLM** — same 1,500-review sample, everything run free on a 4 GB
laptop GPU (Quadro P2000):

| | Stanford MAE | Stanford pos/neg | TMDB MAE | TMDB Spearman |
|---|---|---|---|---|
| DistilBERT fine-tuned (66M params, 22k labels) | **1.10** | **92.2%** | **1.18** | **0.68** |
| Qwen2.5-0.5B-Instruct, zero-shot | 2.07 | 75.0% | 1.28 | 0.61 |
| Qwen2.5-0.5B-Instruct, calibrated on Stanford | 1.85 | 86.1% | 1.84 | 0.61 |

The LLM's score is the probability-weighted mean of its next-token distribution over 1-10
(no output parsing; 99.9% of the probability mass lands on valid scores).

Findings:
- Fine-tuning a small model beats frozen embeddings, which beat bag-of-words, on every set;
  the biggest gain is out of domain (TMDB Spearman 0.50 → 0.58 → 0.69).
- The zero-shot LLM over-rates Stanford's deliberately polarized reviews by 1.4 stars, yet on
  real-world TMDB reviews it is on par with the trained baselines without any training data.
- Calibration is distribution-specific: a linear map fitted on Stanford validation fixes the
  Stanford bias but *hurts* on TMDB (MAE 1.28 → 1.84), because it learns Stanford's scale.
- Engineering: per-epoch checkpoints with full RNG state (a crashed-then-resumed run matches
  an uninterrupted one exactly), fp32 training because Pascal GPUs lack fast fp16, and
  resumable, checksum-verified model downloads.

```bash
uv sync --group ml
uv run python -m imdb_insights.reviews prepare
uv run python -m imdb_insights.reviews train tfidf        # also: embeddings, finetune
uv run python -m imdb_insights.reviews llm               # local zero-shot LLM
uv run python -m imdb_insights.reviews evaluate          # writes reports/review_models.md
```

## AI assistant

Ask questions in plain English on the app's **Ask** page (or `python -m imdb_insights.assistant
ask "..."`). A tool-calling agent picks between two tools:

- **`run_sql`** for facts and numbers: one DuckDB `SELECT` with layered guardrails — DuckDB's
  own parser must classify it as a single SELECT, the connection is read-only, external
  access is disabled (a `SELECT * FROM read_csv('/etc/passwd')` parses as SELECT but is
  refused), and results are capped with a timeout.
- **`search_reviews`** for opinions: hybrid search over 31k review passages — BM25 (DuckDB
  FTS) plus bge-small embeddings (ONNX Runtime, no PyTorch), merged with reciprocal rank
  fusion. Answers cite passages as [R1], [R2]; the app lists only the cited sources.

The model is any OpenAI-compatible endpoint (default: Groq's free tier,
`openai/gpt-oss-120b`). Tool errors and provider-rejected tool calls go back to the model so
it can correct itself; every step is shown under "How I got this".

**Evaluation** (`python -m imdb_insights.assistant eval`, [`reports/assistant_eval.md`](reports/assistant_eval.md)):
21 cases — facts graded against reference SQL run on the live warehouse, opinions graded on
valid citations from the right movie, and safety cases (off-topic, destructive request,
prompt injection).

| model (Groq free tier) | passed | median latency | tokens / question |
|---|---|---|---|
| openai/gpt-oss-120b | 21/21 | 3.3 s | ~2,000 |
| qwen/qwen3.8-27b | 21/21 | 18.4 s | ~2,400 |

Both models saturate this set, so its next version needs harder cases (multi-step questions,
ambiguous titles, unanswerable questions). Grading bugs it surfaced, now fixed: GPT-OSS writes
citations with zero-width spaces (`[\u200bR2\u200b]`) or CJK brackets (`【R1】`).

## Why APIs, not scraping

- IMDb returns `403 Forbidden` to automated browsers and requires sign-in for more than five
  reviews per title. Working around either would mean evading bot detection and breaking
  IMDb's terms of use — not something a production system (or this project) does.
- TMDB and OMDb are built for programmatic access, link to IMDb through IMDb ids, and give
  richer data than the old scraper collected.
- Raw API data stays out of the repository (`data/` is git-ignored); the pipeline rebuilds it.

## Roadmap

- [x] Resumable multi-source ingestion, DuckDB warehouse, IMDb-style ranking, tests, CI
- [x] Streamlit app rebuilt on the new warehouse, with smoke tests for every page
- [x] Deployment pipeline: daily refresh via GitHub Actions, private Hugging Face storage
- [x] Review rating prediction: TF-IDF vs embeddings vs fine-tuned transformer vs local LLM,
      with leakage checks, out-of-domain testing and bootstrap confidence intervals
- [x] AI assistant: text-to-SQL over the warehouse + hybrid retrieval over reviews, with
      citations, guardrails and an evaluation set
- [ ] LLMOps: tracing, evaluation sets, prompt versioning; Docker; MCP server

## Credits

- Data: [TMDB](https://www.themoviedb.org/) and [OMDb](https://www.omdbapi.com/).
  *This product uses the TMDB API but is not endorsed or certified by TMDB.*
- v1 (2024) by Ahmed Abdelmoneim and Mazen Ashraf.
