# Evaluation and revision

The SQL critic evaluates question relevance, approved schema, aggregation and
rule compliance, using separate historical and forecast rulebooks. It receives
schema metadata, not sample business rows. Malformed verdicts get one review-only
repair attempt; rejected SQL also gets an audit before requesting a rewrite.
Reviewer uncertainty never counts as approval. After bounded revisions, SQL with
no deterministic issues may still run through the mandatory database gateway,
with its output explicitly marked as unverified query results.
Forecast caveat checks do not apply to historical questions. Period breakdowns
can contain up to twenty cited statements. The SQLite gateway remains the
independent permission and resource boundary.

After execution, the summary generator receives evidence with stable fact IDs:
filtered row samples, numeric statistics over all filtered result rows, and
forecast metadata/backtest metrics when present. Each summary statement cites
supporting facts internally. Code checks those references and digit-based numeric
values against the cited facts (rounded to two decimal places). The LLM then
checks meaning, units, group attribution, completeness, unsupported claims and
forecast caveats. Numerical membership alone does not prove a claim is correct;
semantic review is still probabilistic. Written-out numbers and causal claims
are evaluated by the LLM, not the numeric matcher.

Failed reviews target either SQL or summary generation. SQL revisions pass through
the critic and gateway again and clear stale forecast state. Set
`MAX_EVALUATION_RETRIES=2` in `.env` to change the shared post-answer revision
budget (0-4). SQL and forecast attempts retain their own existing budgets. The
graph recursion limit accounts for all bounded loops. Failed or unavailable
evaluation withholds the AI narrative and forecast payload, skips charts, and
shows successfully retrieved rows under `data_only` status. These rows are not
claimed to be a verified answer. This response is not cached. Safety violations,
missing SQL, deterministic rule failures, and execution errors still block output.
This adds model calls and latency.

Simple unfiltered monthly forecasts of orders or revenue use a conservative,
schema-checked historical SQL template. The requested future year is handled by
the statistical engine, not used to filter history. Unknown words, group filters,
other metrics and other frequencies fall back to general SQL generation. For
these verified history plans, a successful statistical forecast remains visible
with deterministic caveats even when AI narrative review fails. `passed` stays
false for the narrative; the forecast is not represented as actual future data.

Responses include `sql_evaluation`, `evaluation` and `evaluation_history`.
Verdicts contain boolean checks, issues, a revision target and fix instructions.
Final answer status is `passed`, `data_only`, or `blocked`; these are runtime checks, not a
guarantee of correctness or a forecast accuracy percentage. Charts are generated
after the answer passes and are not covered by the answer evaluator.

## Repeatable evaluation

From this directory:

```powershell
python -m pytest -q
python run_evals.py
python run_evals.py --live
```

`evals/cases.json` contains synthetic SQLite rows, questions, reference SQL,
expected ordered result values, and positive/negative numeric claim cases.
The offline command checks reference results and numerical guardrails, not LLM
quality. `--live` runs actual question-to-answer generation against a temporary
fixture database and compares returned values plus evaluation acceptance. It
uses your configured model/API key and incurs usage; it never queries your real
database. Execute it as a separate CLI process. Outputs are JSON reports with
per-case pass/fail results and a nonzero exit code on regression. Column aliases
are ignored, but selected column order and row order must match the gold data.

This is a starter dataset, not a production accuracy benchmark. Extend it with
your business questions, joins, empty results, ambiguous requests and representative
forecast series. The current live cases cover historical queries; forecast
evidence and routing have automated regression coverage, but there is no live
forecast quality benchmark yet. Record model/version and compare reports across
changes; never treat a judge agreeing with itself as ground truth.
