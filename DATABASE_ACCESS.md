# SQLite access policy

Keep the existing `DB_FILE` (or `DATASAGE_DB_FILE`) path in `.env`. No migration
is needed. The SQL generator receives approved schema metadata only, has no
database tools, and sends candidate SQL through critic review and the gateway.
After exhausted critic retries, a query with no deterministic issues may run
through the gateway to show unverified results without AI analysis. Reviewer
approval is never fabricated; permission and resource checks still apply.

Optional settings (defaults shown):

```dotenv
DB_ALLOWED_TABLES=orders
# Empty means all columns on approved tables. To restrict, use table.column:
# DB_ALLOWED_COLUMNS=orders.order_date,orders.region,orders.total_sales
DB_ALLOWED_COLUMNS=
DB_QUERY_TIMEOUT_SECONDS=5
SQL_ROW_LIMIT=100
DB_FORECAST_ROW_LIMIT=10000
DB_MAX_RESULT_BYTES=2097152
DB_MAX_CONCURRENT_QUERIES=4
```

Restart the app after changing policy. The table and column policy is enforced
by SQLite's authorizer, including columns used in filters, joins, and ordering.
COUNT(*) / row-existence queries are permitted on approved tables. Functions
use an explicit allowlist; denied functions require a reviewed code change.
Views and virtual-table access are not enabled by the default policy.

Queries use read-only connections. A progress handler interrupts SQL when its
time budget expires. Lock waiting is separately bounded. Oversized results fail
with a scope-narrowing message; forecast history is never silently truncated.
Concurrency is limited per gateway/application process. Result byte limits
bound serialized output, not total SQLite or Python process memory. These are
cooperative query deadlines, not operating-system CPU/memory isolation; use
isolated workers for hard process limits.

Permissions apply to the whole application, not individual users or tenant rows.
Only include data this application's users may access. `ALLOWED_LLM_COLUMNS`
continues to filter output fields sent to the summary/chart models; it is not
a source-column permission boundary. Use `DB_ALLOWED_COLUMNS` to prevent reads
of sensitive source columns, including reads through aliases and expressions.
No business-row samples are sent with the SQL schema.

Run `python -m pytest -q` from this directory. Gateway tests use temporary
SQLite fixtures and mocked model calls; they need neither API keys nor real data.
