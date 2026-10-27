# DataSage BI Assistant

DataSage is a Streamlit app for asking questions about a SQLite database. It generates SQL, reviews it, executes it through a read-only gateway, and presents the results with a written summary and, when possible, a chart. Forecast questions use historical SQL results to run a local statistical forecasting workflow.

## Requirements

- Python 3.11 (the version used by CI and the Docker image) or Docker Desktop with its Linux engine running
- An OpenAI API key for the interactive app
- A SQLite database with tables and columns you intend to expose to the app

## Run locally

From the repository directory, create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. Create a `.env` file in the repository directory; there is currently no `.env.example` to copy:

```dotenv
OPENAI_API_KEY=your_api_key
DB_FILE=C:/path/to/your/database.db
DB_ALLOWED_TABLES=orders
```

Use the actual database path and list every table the app should be allowed to read, separated by commas. `DB_ALLOWED_TABLES` defaults to `orders`. On Windows, forward slashes in `DB_FILE` avoid backslash escaping issues.

Start the app with `python -m streamlit run app.py`, then open the URL printed by Streamlit (normally `http://localhost:8501`). Restart it after changing environment settings or the database schema.

## Run with Docker Compose

Create `.env.docker` with `OPENAI_API_KEY=...`. Edit the `source` bind-mount path in [compose.yaml](compose.yaml) to point to your SQLite file. The container reads that file at `/app/data/superstore.db`; Compose sets `DB_FILE` to this container path. If you change the container filename, update both the mount target and `DB_FILE`. Add `DB_ALLOWED_TABLES` to `.env.docker` if the allowed tables are not just `orders`.

```powershell
docker compose up -d --build
docker compose logs -f datasage
```

Open `http://localhost:8501`. Use `docker compose down` to stop the container. If Compose reports that `dockerDesktopLinuxEngine` or `docker_engine` cannot be found, start Docker Desktop and wait for its Linux engine before retrying. The image is built locally; this Compose file does not publish it to a registry.

The current Compose file exposes port 8501 on the host. For a shared or networked deployment, set `AUTH_REQUIRED=1` and `DATASAGE_APP_PASSWORD` in `.env.docker` before exposing the app. Authentication is off by default. Keep `.env`, `.env.docker`, and private database files out of Git; they are ignored by [.gitignore](.gitignore). If a credential was previously committed, remove it from tracking and rotate it; adding an ignore rule does not remove it from Git history.

## Database access and schema

The gateway discovers metadata for approved SQLite tables and enforces table/column permissions, read-only queries, row and result-size limits, a query deadline, and a per-process concurrency limit. Set `DB_ALLOWED_COLUMNS` to a comma-separated list of `table.column` entries when all columns of an approved table should not be readable. `ALLOWED_LLM_COLUMNS` only filters result fields sent to models; it is not a database permission rule. See [DATABASE_ACCESS.md](DATABASE_ACCESS.md) for the settings and limits.

The app supports multiple approved tables, but the SQL generation prompts and some fast-path query templates are tailored to the `orders` schema. For that schema, `order_date` drives time-based queries, `total_sales` is the sales/revenue measure, and `order_id` is used for distinct order counts. Queries against other schemas use a more general prompt; review the business meaning of generated SQL and add representative evaluation cases when changing datasets. Merely changing `DB_FILE` does not establish that a new dataset's metrics are interpreted correctly.

## Answers, forecasts, and usage

SQL and narrative reviews can trigger revisions. A passed review is an automated check, not a guarantee of business correctness. When review cannot verify an answer, the UI distinguishes qualified answers from data-only or blocked results. Historical forecast error is measured by backtesting; it is not a guaranteed future accuracy rate. Forecast charts are built from the statistical result, while historical-result charts may be generated alongside summary review. See [EVALUATION.md](EVALUATION.md) for the review and forecast behavior.

The **Show request details** button reveals latency, model calls, reported tokens, and estimated API cost. The app also writes request metrics to its process logs (`docker compose logs` when using Compose). Cost estimates exclude unreported SDK retries and infrastructure charges. The default model is `gpt-4o-mini`; set `OPENAI_MODEL` to change it.

## Tests and CI

```powershell
python -m pytest -q
python run_evals.py
```

These checks use local fixtures and do not require a live OpenAI request. `python run_evals.py --live` exercises the model against a temporary fixture database and incurs API usage. GitHub Actions runs syntax checks, tests, offline evaluations, and a Docker build on pushes to any branch and pull requests targeting `main`. The Docker build checks the image but does not deploy it.
