# IMDb Insights AI

An end-to-end data and AI project around the best-rated movies of all time: a resumable,
API-based ingestion pipeline, an analytical DuckDB warehouse, an IMDb-style Top 250 we compute
ourselves, and (in progress) NLP over reviews and an AI assistant.

> **v2** of [IMDb-Top-250-Scraper-Review](https://github.com/Ahmed-Abdellmoneim/IMDb-Top-250-Scraper-Review)
> (2024). IMDb has since blocked automated browsers and put reviews behind sign-in, so v2
> collects data through official APIs instead of scraping — see [Why APIs](#why-apis-not-scraping).

## What it does today

```
TMDB discover ──► candidates (rating ≥ 7.3, ≥ 600 votes)
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
- **Incremental.** Cached responses expire after ~7 days (TMDB) and ~14 days (OMDb), spread
  per key so refreshes are distributed across days and stay inside OMDb's daily quota.
  Movies with no rating at all are fetched first; if a refresh fails, the last known value
  is kept and counted in the manifest.
- **Zero-downtime refresh.** The app polls for a new warehouse at most hourly and swaps it in
  atomically; a failed download keeps the current data.

Configuration: repository secrets `TMDB_API_KEY`, `OMDB_API_KEY`, `HF_TOKEN` (write) and the
variable `HF_DATASET_REPO`; the Streamlit app needs `HF_TOKEN` (read-only) and `HF_DATASET_REPO`.

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
- [ ] Review rating prediction: embeddings baseline vs fine-tuned transformer vs LLM, with
      proper evaluation (Stanford Large Movie Review Dataset + TMDB reviews)
- [ ] AI assistant: text-to-SQL over the warehouse + retrieval over reviews, with citations
- [ ] LLMOps: tracing, evaluation sets, prompt versioning; Docker; MCP server

## Credits

- Data: [TMDB](https://www.themoviedb.org/) and [OMDb](https://www.omdbapi.com/).
  *This product uses the TMDB API but is not endorsed or certified by TMDB.*
- v1 (2024) by Ahmed Abdelmoneim and Mazen Ashraf.
