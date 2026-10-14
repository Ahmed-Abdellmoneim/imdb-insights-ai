import duckdb
import pytest

from imdb_insights.assistant import sql


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "w.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("create table movies as select range as tmdb_id, 'Movie ' || range as title "
                    "from range(100)")  # fmt: skip
    return path


def test_select_runs_and_is_capped(db):
    result = sql.run_query(db, "select * from movies order by tmdb_id", max_rows=10)
    assert len(result.frame) == 10 and result.truncated
    assert result.to_text().splitlines()[0] == "tmdb_id,title"


def test_ctes_and_trailing_semicolon_are_fine(db):
    result = sql.run_query(db, "with x as (select count(*) as n from movies) select n from x;")
    assert result.frame["n"].iloc[0] == 100 and not result.truncated


@pytest.mark.parametrize(
    "query",
    [
        "drop table movies",
        "delete from movies",
        "insert into movies values (1, 'x')",
        "copy movies to 'out.csv'",
        "attach 'other.duckdb'",
        "install httpfs",
        "select 1; drop table movies",
        "set enable_external_access = true",
    ],
)
def test_non_select_statements_are_rejected(db, query):
    with pytest.raises(sql.UnsafeQueryError):
        sql.run_query(db, query)


@pytest.mark.parametrize(
    "query",
    [
        "select * from read_csv('{p}')",
        "select * from '{p}'",
        "select * from read_text('{p}')",
    ],
)
def test_file_access_is_blocked_even_inside_select(db, tmp_path, query):
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n", encoding="utf-8")
    with pytest.raises(duckdb.Error):
        sql.run_query(db, query.format(p=secret.as_posix()))


def test_database_is_not_modified(db):
    with pytest.raises(sql.UnsafeQueryError):
        sql.run_query(db, "update movies set title = 'x'")
    assert sql.run_query(db, "select count(*) as n from movies").frame["n"].iloc[0] == 100


def test_slow_queries_time_out(db, monkeypatch):
    monkeypatch.setattr(sql, "TIMEOUT_SECONDS", 0.2)
    with pytest.raises(TimeoutError):
        sql.run_query(db, "select count(*) from range(1000000000) a, range(1000) b")


def test_unparseable_sql_is_reported():
    with pytest.raises(sql.UnsafeQueryError, match="parse"):
        sql.check_query("selec * form movies")
