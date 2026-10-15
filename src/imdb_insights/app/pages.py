"""The app's pages, one function each (wired up in ``main.py``)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import charts, data

IMDB_TITLE_URL = "https://www.imdb.com/title/{}/"
TMDB_MOVIE_URL = "https://www.themoviedb.org/movie/{}"


def _fmt_int(value) -> str:
    return "—" if pd.isna(value) else f"{int(value):,}"


def _fmt(value, suffix: str = "") -> str:
    return "—" if value is None or pd.isna(value) else f"{value}{suffix}"


def completeness_banner() -> None:
    quality = data.manifest().get("quality", {})
    if quality and not quality.get("complete", True):
        pending = quality.get("omdb_status", {}).get("pending", 0)
        st.warning(
            f"The ranking is provisional: {pending:,} candidate movies are still waiting for "
            "their IMDb rating (OMDb's free quota is 1,000 requests a day).",
            icon=":material/hourglass_top:",
        )


# --- Top 250 --------------------------------------------------------------------------


def top250_page() -> None:
    st.title("Top 250 movies")
    st.caption(
        "Ranked with IMDb's weighted-rating formula over IMDb ratings and vote counts. "
        "See *About the data* for the method."
    )
    completeness_banner()
    movies = data.top250()

    all_genres = sorted({g for gs in movies["genres"] for g in gs})
    years = movies["year"].dropna().astype(int)
    col1, col2, col3 = st.columns([2, 2, 2])
    genres = col1.multiselect("Genres", all_genres, placeholder="All genres")
    year_range = None
    if years.nunique() > 1:  # a slider needs two distinct ends
        lo, hi = int(years.min()), int(years.max())
        year_range = col2.slider("Release year", lo, hi, (lo, hi))
    search = col3.text_input("Title or director", placeholder="e.g. Nolan")

    view = movies
    if year_range and year_range != (years.min(), years.max()):
        view = view[view["year"].between(*year_range)]
    if genres:
        view = view[view["genres"].map(lambda gs: bool(set(gs) & set(genres)))]
    if search:
        needle = search.casefold()
        view = view[
            view["title"].str.casefold().str.contains(needle, regex=False)
            | view["directors"].map(lambda ds: any(needle in d.casefold() for d in ds))
        ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies shown", f"{len(view):,}")
    c2.metric("Average IMDb rating", f"{view['imdb_rating'].mean():.2f}" if len(view) else "—")
    c3.metric("IMDb votes", f"{view['imdb_votes'].sum() / 1e6:,.1f}M")
    c4.metric("Oscars won", f"{int(view['oscar_wins'].fillna(0).sum()):,}")

    table = view.assign(
        imdb_url=view["imdb_id"].map(lambda i: IMDB_TITLE_URL.format(i) if i else None),
        genres=view["genres"].map(", ".join),
        directors=view["directors"].map(", ".join),
    )
    st.dataframe(
        table[
            [
                "rank",
                "poster_thumb_url",
                "title",
                "year",
                "imdb_rating",
                "imdb_votes",
                "weighted_rating",
                "metascore",
                "genres",
                "directors",
                "imdb_url",
            ]
        ],
        hide_index=True,
        width="stretch",
        height=640,
        row_height=72,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "poster_thumb_url": st.column_config.ImageColumn("", width="small"),
            "title": st.column_config.TextColumn("Title", width="medium"),
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "imdb_rating": st.column_config.NumberColumn("IMDb", format="%.1f"),
            "imdb_votes": st.column_config.NumberColumn("Votes", format="localized"),
            "weighted_rating": st.column_config.NumberColumn(
                "Score", format="%.3f", help="Weighted rating used for the ranking"
            ),
            "metascore": st.column_config.NumberColumn("Metascore"),
            "genres": "Genres",
            "directors": "Director",
            "imdb_url": st.column_config.LinkColumn("IMDb page", display_text="Open"),
        },
    )


# --- Movie ----------------------------------------------------------------------------


def movie_page() -> None:
    movies = data.top250()
    labels = {
        row.tmdb_id: f"#{row.rank} · {row.title} ({_fmt(row.year)})" for row in movies.itertuples()
    }
    default = st.query_params.get("id")
    ids = list(labels)
    index = (
        ids.index(int(default)) if default and default.isdigit() and int(default) in labels else 0
    )
    tmdb_id = st.selectbox("Movie", ids, index=index, format_func=labels.get)
    st.query_params["id"] = str(tmdb_id)
    m = data.movie(tmdb_id)

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        if m.get("poster_url"):
            st.image(m["poster_url"], width="stretch")
        links = [
            f"[IMDb]({IMDB_TITLE_URL.format(m['imdb_id'])})" if m.get("imdb_id") else None,
            f"[TMDB]({TMDB_MOVIE_URL.format(tmdb_id)})",
        ]
        st.markdown(" · ".join(link for link in links if link))
    with right:
        st.title(m["title"])
        facts = [
            _fmt(m.get("year")),
            m.get("rated"),
            f"{int(m['runtime_minutes'])} min" if pd.notna(m.get("runtime_minutes")) else None,
            ", ".join(m.get("genres") or []),
        ]
        st.caption(" · ".join(f for f in facts if f and f != "—"))
        if m.get("tagline"):
            st.markdown(f"*{m['tagline']}*")
        st.write(m.get("overview") or "")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("IMDb", _fmt(m.get("imdb_rating")), help=f"{_fmt_int(m.get('imdb_votes'))} votes")
        r2.metric(
            "TMDB",
            f"{m['tmdb_rating']:.1f}" if pd.notna(m.get("tmdb_rating")) else "—",
            help=f"{_fmt_int(m.get('tmdb_votes'))} votes",
        )
        r3.metric("Metascore", _fmt(m.get("metascore")))
        r4.metric("Rotten Tomatoes", _fmt(m.get("rotten_tomatoes"), "%"))

        crew = data.key_crew(tmdb_id)
        for row in crew.itertuples():
            st.markdown(f"**{row.job}:** {', '.join(row.names)}")
        money = [("Budget", m.get("budget_usd")), ("Worldwide revenue", m.get("revenue_usd"))]
        # "\$" stops Streamlit's markdown from reading "$...$" as a LaTeX formula.
        money_text = [f"**{k}:** \\${int(v):,}" for k, v in money if pd.notna(v)]
        if money_text:
            st.markdown(" · ".join(money_text))
        if m.get("awards"):
            st.markdown(f"**Awards:** {m['awards']}")

    st.subheader("Cast")
    cast = data.cast(tmdb_id)
    cols = st.columns(6)
    for i, person in enumerate(cast.itertuples()):
        with cols[i % 6]:
            if person.profile_url:
                st.image(person.profile_url, width="stretch")
            st.markdown(f"**{person.name}**  \n{person.character or ''}")

    kws = data.keywords(tmdb_id)
    if kws:
        st.subheader("Keywords")
        st.pills("Keywords", kws, label_visibility="collapsed", disabled=True)

    reviews = data.reviews(tmdb_id)
    st.subheader(f"Reviews ({len(reviews)})")
    if reviews.empty:
        st.caption("No TMDB reviews for this movie yet.")
    for r in reviews.itertuples():
        rating = f" · {r.rating:g}/10" if pd.notna(r.rating) else ""
        date = r.created_at.date().isoformat() if pd.notna(r.created_at) else ""
        with st.expander(f"{r.author or 'Anonymous'}{rating} · {date}"):
            st.write(r.content)
            if r.url:
                st.markdown(f"[Read on TMDB]({r.url})")


# --- Insights -------------------------------------------------------------------------


def insights_page() -> None:
    st.title("Insights")
    completeness_banner()

    st.subheader("Which decades made the list?")
    decades = data.by_decade()
    decades["decade_label"] = decades["decade"].astype(int).astype(str) + "s"
    charts.show(
        charts.vbar(
            decades,
            "decade_label",
            "films",
            x_title=None,
            y_title="Films in the Top 250",
            hover="%{x}<br>%{y} films<br>Avg IMDb rating %{customdata[2]}",
        ),
        decades[["decade_label", "films", "avg_imdb_rating"]],
        key="decades",
    )

    st.subheader("Genres")
    genres = data.genre_stats()
    st.caption("A movie counts once for each of its genres.")
    charts.show(
        charts.hbar(
            genres,
            "films",
            "genre",
            value_title="Films in the Top 250",
            hover="%{y}<br>%{x} films<br>Avg IMDb rating %{customdata[2]}",
        ),
        genres,
        key="genres",
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Directors with the most films")
        directors = data.top_people("Director")
        charts.show(
            charts.hbar(
                directors,
                "films",
                "name",
                value_title="Films",
                hover="%{y}<br>%{x} films<br>%{customdata[3]}",
                height=460,
            ),
            directors,
            key="directors",
        )
    with right:
        st.subheader("Most-billed actors")
        st.caption("Counting only the top five billed roles of each film.")
        actors = data.top_people("Actor", max_billing=5)
        charts.show(
            charts.hbar(
                actors,
                "films",
                "name",
                value_title="Films",
                hover="%{y}<br>%{x} films<br>%{customdata[3]}",
                height=460,
            ),
            actors,
            key="actors",
        )

    ratings = data.ratings_comparison()
    st.subheader("Do IMDb and TMDB audiences agree?")
    both = ratings.dropna(subset=["imdb_rating", "tmdb_rating"])
    gap = (both["imdb_rating"] - both["tmdb_rating"]).mean()
    varied = both["imdb_rating"].nunique() > 1 and both["tmdb_rating"].nunique() > 1
    corr = both["imdb_rating"].corr(both["tmdb_rating"]) if varied else float("nan")
    a, b = st.columns(2)
    a.metric("Average gap (IMDb − TMDB)", "—" if pd.isna(gap) else f"{gap:+.2f} points")
    b.metric("Correlation", "—" if pd.isna(corr) else f"{corr:.2f}")
    charts.show(
        charts.scatter(
            both,
            "tmdb_rating",
            "imdb_rating",
            x_title="TMDB rating",
            y_title="IMDb rating",
            diagonal=True,
            hover="%{customdata[0]} (%{customdata[1]})<br>IMDb %{y} · TMDB %{x:.2f}",
        ),
        both[["title", "year", "imdb_rating", "tmdb_rating"]],
        key="imdb_vs_tmdb",
    )
    st.caption("Points above the grey line are rated higher on IMDb than on TMDB.")

    st.subheader("Critics vs audiences")
    critics = ratings.dropna(subset=["imdb_rating", "metascore"])
    charts.show(
        charts.scatter(
            critics,
            "metascore",
            "imdb_rating",
            x_title="Metascore (critics, 0–100)",
            y_title="IMDb rating (audience)",
            hover="%{customdata[0]} (%{customdata[1]})<br>Metascore %{x} · IMDb %{y}",
        ),
        critics[["title", "year", "imdb_rating", "metascore"]],
        key="critics",
    )

    st.subheader("Budget and box office")
    money = data.money()
    st.caption(
        f"{len(money)} films with a known budget and revenue (TMDB). Both axes are logarithmic; "
        "points above the grey line earned more than they cost."
    )
    charts.show(
        charts.scatter(
            money,
            "budget_usd",
            "revenue_usd",
            x_title="Budget (USD)",
            y_title="Worldwide revenue (USD)",
            diagonal=True,
            log=True,
            hover="%{customdata[0]} (%{customdata[1]})<br>Budget $%{x:,.0f}<br>"
            "Revenue $%{y:,.0f}<br>%{customdata[4]}× budget",
        ),
        money,
        key="money",
    )


# --- About ----------------------------------------------------------------------------


def about_page() -> None:
    st.title("About the data")
    manifest = data.manifest()
    quality = manifest.get("quality", {})
    completeness_banner()

    pools = quality.get("candidate_pools") or []
    pool = (
        "Every TMDB movie " + ", plus every movie ".join(pools) + " is a candidate"
        if pools
        else "The best-rated TMDB movies are candidates"
    )
    st.markdown(
        f"""
        **Sources.** Candidate movies, details, cast & crew, keywords and reviews come from the
        [TMDB API](https://www.themoviedb.org/). IMDb ratings and vote counts, Metascore,
        Rotten Tomatoes scores and award summaries come from the [OMDb API](https://www.omdbapi.com/).
        The two are joined on each movie's IMDb id.

        **Ranking.** {pool} (the floors sit well below the weakest ranked movie, because TMDB
        users rate lower than IMDb users and some films have far fewer TMDB votes). Candidates
        with at least 25,000 IMDb votes are ranked by IMDb's published weighted rating:
        """
    )
    st.latex(r"WR = \frac{v}{v+m}R + \frac{m}{v+m}C \qquad m = 25{,}000,\ C = 7.0")
    st.markdown(
        "where *R* is the movie's IMDb rating and *v* its vote count. Movies with fewer votes "
        "are pulled towards *C*, so a handful of enthusiastic votes can't top the chart."
    )

    if manifest:
        st.subheader("Latest run")
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates", f"{quality.get('candidates', 0):,}")
        c2.metric("Ranked", f"{quality.get('ranked', 0):,}")
        c3.metric("Complete", "Yes" if quality.get("complete") else "Not yet")
        st.caption(f"Finished {manifest.get('finished_at', '')[:16].replace('T', ' ')} UTC")
        rows = pd.DataFrame([{"table": k, "rows": v} for k, v in manifest.get("rows", {}).items()])
        st.dataframe(rows, hide_index=True)
        for warning in quality.get("warnings", []):
            st.warning(warning)

    st.divider()
    st.caption(
        "This product uses the TMDB API but is not endorsed or certified by TMDB. "
        "IMDb ratings via OMDb. Not affiliated with IMDb."
    )
    if version := _code_version():
        st.caption(f"App version {version}")


@st.cache_resource
def _code_version() -> str | None:
    """The deployed git commit, read from the checkout's .git folder (no git binary needed)."""
    from pathlib import Path

    for folder in (Path.cwd(), *Path(__file__).resolve().parents):
        git = folder / ".git"
        if not (git / "HEAD").is_file():
            continue
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head[:7]  # detached HEAD: the commit itself
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip()[:7]
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line[:7]
        return None
    return None
