# SQL review correctness investigation

## Scope and method

This investigation used 16 fixed questions: eight detail lookups (first, latest,
and month-scoped orders) and eight grouped forecasts (region or category;
sales, revenue, or order count). The exact synthetic questions and fixture
generator are in `investigate_sql_review.py`. The local SQLite fixture has
36 consecutive months and two values for each grouping dimension. No
production database or traffic was used. Raw questions, candidate SQL, and
verdicts were kept in ignored local `.aidebug/critic_*.json` files, not in
production logs or this report.

The baseline and first rerun each made 16 live requests against the same
fixture, question order, `gpt-4o-mini` reviewer, retry settings, and no answer
cache. The second, targeted rerun made eight grouped-forecast requests with
`gpt-4o` only for forecast SQL review. Counts below refer to *candidate SQL
reviews*, not merely HTTP calls: a reviewer can make a second internal model
call to repair a malformed verdict or audit a content rejection.

Candidate results were checked locally through `DatabaseGateway` against
reference results and the question's requested sort, filter, dimension, and
metric. For the eight first grouped-forecast candidates, the full 72-row
historical result matched the reference in each phase. Five of eight first
lookup candidates matched the reference and obeyed the requested ordering;
three first-five queries omitted `ORDER BY`. They happened to return the same
five rows on this fixture, but did not guarantee the requested *first* five.
This semantic error was treated as a genuine generation defect.

## Diagnosis

| Pattern | Observed cause | Ground-truth finding |
|---|---|---|
| First-five lookups | SQL generation omitted ordering on three first candidates. | The critic was right to object to an undefined first-five selection. |
| Detail lookups with a sales column | The model sometimes named `total_sales` as `requested_metric_column` despite the prompt specifying null for detail questions. `metric_sql_issue()` then demanded that column inside an aggregate. | A raw projection of the requested metric is correct for a detail-row question. This deterministic check caused false rejections. |
| Other lookups | The mini reviewer produced malformed or internally contradictory verdicts; it also sometimes demanded aggregation for a valid detail query. | These are verdict-format and content-judgment problems, respectively, separate from the metric check. |
| Grouped forecasts | The mini reviewer repeatedly treated a valid historical aggregate as a future prediction, or demanded a history `WHERE` or SQL forecast horizon. | The candidate SQL was correct historical training data with the requested grouping, metric, and all available periods. This was an LLM content-judgment false rejection. No deterministic pre-check was responsible. |

The reviewer’s two repair paths were counted separately. A **format repair**
follows an invalid or contradictory JSON verdict. An **audit repair** follows
a validly formed content rejection and asks the reviewer to reassess it.
Neither path by itself proves the SQL was wrong.

## Matched-model comparison

The first rerun includes the detail projection fix and clearer SQL generator
and reviewer instructions, but retains `gpt-4o-mini` for both categories.

| Category and phase | SQL candidate reviews | Approved candidate verdicts | Format repairs | Audit repairs | Final outcomes | Median wall time |
|---|---:|---:|---:|---:|---|---:|
| Lookup, before | 16 | 10 | 7 | 5 | 1 passed, 2 data_only, 5 blocked | 7.63 s |
| Lookup, after | 11 | 6 | 4 | 1 | 6 passed, 2 data_only | 6.35 s |
| Grouped forecast, before | 16 | 0 | 0 | 16 | 8 data_only | 9.01 s |
| Grouped forecast, after | 15 | 1 | 0 | 15 | 8 data_only | 9.49 s |

“Approved candidate verdicts” counts individual model verdicts, including
ones on later SQL attempts; it is not the request pass rate. The lookup result
is materially better but not complete. One remaining lookup had missing
ordering and contradictory verdicts; another valid detail query was still
falsely rejected as requiring aggregation. The grouped result shows that
prompt wording alone did **not** fix the false-rejection pattern. Wall-time
differences are not correctness measures and this small live sample is subject
to API variability.

## Targeted forecast reviewer intervention

Because the same-model grouped rerun still rejected almost every valid
historical query, the forecast SQL critic now uses a separately configurable
`FORECAST_SQL_CRITIC_MODEL`, defaulting to `gpt-4o`. Historical SQL review and
answer review retain the existing model. This is a separate intervention, so
its latency and cost are **not** part of the matched-model comparison above.
The SQL critic and its checks remain mandatory; no SQL fast path or retry
budget changed.

The same eight grouped questions with this reviewer had **8/8 first candidate
approvals**, 10/10 candidate approvals overall (two later SQL revisits), no
format or audit repairs, and all eight reached forecast fitting and
backtesting. All eight still ended `data_only` because downstream answer
verification did not approve the narrative. SQL approval therefore does not
establish that the end-to-end forecast answer is fixed. Median wall time was
18.00 s, including forecast and answer stages that the earlier rejected runs
never reached. Estimated API cost for these eight requests was $0.05250,
versus $0.01202 for the eight mini-reviewer grouped requests; this is not a
like-for-like successful-output cost comparison.

The actual snapshot returned by the API was `gpt-4o-2024-08-06`. Its estimated
price is recorded in `telemetry.py` using the [official model pricing](https://developers.openai.com/api/docs/models/gpt-4o).
The same official page currently marks GPT-4o as deprecated, so the model
choice should be revisited before a long-lived deployment. The environment
setting permits replacing it without changing the SQL review pipeline.

## Changes and verification

| File | Change | Rejection cause addressed |
|---|---|---|
| `query_planning.py` | Accept a requested metric projected directly in non-aggregate detail SQL; still check aggregate references. | False deterministic rejection of detail rows. |
| `bi.py` | Clarify detail versus total-sales SQL; route only forecast SQL review to its configured model. | Wrong detail aggregation guidance; unreliable mini grouped-forecast judgment. |
| `evaluation.py` | Clarify detail validity and historical grouped forecast SQL, with valid and invalid contrasts. | Ambiguous reviewer instructions. |
| `config.py` | Add `FORECAST_SQL_CRITIC_MODEL` without changing retry budgets. | Isolate the reviewer change to forecast SQL. |
| `telemetry.py` | Price the tested GPT-4o alias and snapshot. | Keep per-request cost estimates available. |
| `investigate_sql_review.py` | Save the fixed question set and reproduce the bounded synthetic run. | Comparable baseline and reruns. |
| `tests/` | Cover valid detail projection, incorrect metric/aggregation, missing grouped dimension, future filter, changed scope, model routing, and malformed verdicts. | Check both rejection rigor and verdict-format handling. |

`pytest -q` passed **222 tests**. `run_evals.py` passed **10/10** offline
reference cases. Tests check the positive and negative SQL patterns without
relying on live model behavior. The live sample additionally showed the
stronger reviewer approving correct grouped history, while a five-query
canary rejected wrong metric, lost grouping, future filtering, and a hidden
region filter. These observations do not establish a production-wide rate;
the fixture is small, the model is nondeterministic, and production questions
and schemas may differ.
