# Assistant traces

## By model and prompt version

| model | prompt_version | questions | p50_s | p95_s | avg_tokens | pct_with_tool_error | pct_stopped_early | pct_no_tools | first | last |
|---|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-oss-120b | 045e4ffa08 | 106 | 4.9 | 17.5 | 2768.0 | 1.9 | 0.0 | 18.9 | 2026-11-03 13:22:21 | 2026-11-03 13:38:20 |
| openai/gpt-oss-120b | c67bae9f1b | 96 | 4.1 | 20.7 | 2313.0 | 2.1 | 0.0 | 12.5 | 2026-11-03 13:10:07 | 2026-11-03 13:20:55 |
| openai/gpt-oss-120b | c27b9475a6 | 34 | 5.4 | 18.8 | 2225.0 | 5.9 | 0.0 | 14.7 | 2026-10-15 19:21:02 | 2026-10-15 19:25:42 |
| openai/gpt-oss-120b | 7ac6bf699f | 32 | 5.6 | 19.0 | 2315.0 | 0.0 | 0.0 | 15.6 | 2026-10-15 19:16:14 | 2026-10-15 19:20:10 |
| openai/gpt-oss-120b | e9360b0eed | 32 | 10.2 | 19.2 | 2006.0 | 3.1 | 0.0 | 15.6 | 2026-10-15 19:09:21 | 2026-10-15 19:14:28 |

## Tools

| tool | calls | p50_ms | pct_error |
|---|---|---|---|
| run_sql | 223 | 32.0 | 3.1 |
| search_reviews | 100 | 141.0 | 0.0 |

## Most common tool errors

| tool | error | n |
|---|---|---|
| run_sql | BinderException: Binder Error: Referenced table "m" not found! Candidate tables: "d" LINE 8: SELECT d.person_id, COUNT(* | 3 |
| run_sql | BinderException: Binder Error: Table "c" does not have a column named "role" Candidate bindings: : "credit_type", "job"  | 1 |
| run_sql | BinderException: Binder Error: Referenced table "m" not found! Candidate tables: "df" LINE 10: SELECT p.name, COUNT(*) A | 1 |
| run_sql | BinderException: Binder Error: Referenced table "p" not found! Candidate tables: "people" LINE 1: SELECT * FROM (SELECT  | 1 |
| run_sql | BinderException: Binder Error: Referenced table "m" not found! Candidate tables: "df" LINE 11: ROUND(AVG(m.imdb_rating): | 1 |
