DataSage BI Assistant

DataSage is a Streamlit-based business intelligence assistant that converts natural-language questions into SQL, reviews and executes the query, summarizes the results, and generates visualizations. Forecast-related questions are routed through a dedicated forecasting workflow with historical backtesting.

Features

Natural-language querying over a SQLite database

SQL generation with critic review and retry handling

Read-only SQL safeguards

Business-facing summaries

Vega-Lite visualizations

Forecast-intent routing

Single-series and grouped forecasts

Confidence intervals and historical forecast-error metrics

Project structure

<img width="1000" height="700" alt="ChatGPT Image Sep 8, 2026, 07_44_32 PM" src="https://github.com/user-attachments/assets/ec6fd2cd-bd5d-4ed0-944f-5a7be0c5d666" />


Requirements

Python 3.10 or newer

An OpenAI API key

A SQLite database containing an orders table (see Expected database schema)

Setup

Clone the repository and enter its directory:

git clone https://github.com/YOUR_USERNAME/datasage-bi-assistant.git
cd datasage-bi-assistant

Create and activate a virtual environment.

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS or Linux:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Copy .env.example to .env and provide your local values:

OPENAI_API_KEY=your_openai_api_key
DB_FILE=path/to/superstore.db

On Windows, use forward slashes in the database path:

DB_FILE=C:/Users/your-name/path/to/superstore.db

Do not commit .env or a private database.

Run the application

streamlit run app.py

Run tests

pytest -q test_accuracy.py

Configuration

Variable

Purpose

OPENAI_API_KEY

Authenticates OpenAI model requests

DB_FILE

Filesystem path to the local SQLite database

bi.py validates the database path and derives the SQLAlchemy SQLite URI from it, ensuring the agent toolkit and direct SQL executor use the same database.

Expected database schema

The SQL agent is prompt-tuned to a specific orders table schema. If the database uses different column names, generated queries may fail or may not reflect the intended business meaning.

At minimum, orders should include:

order_date — used for time-based aggregations and forecasting

total_sales — used for revenue and sales calculations

For grouped breakdowns and forecasts, the agent recognizes these dimension columns when present:

region

category

sub_category

segment

state

city

ship_mode

customer_id

If your schema uses different names, rename the columns or adjust BASE_SQL_SYSTEM_PROMPT, FORECAST_SQL_ADDENDUM, and _GROUP_KEYWORDS in bi.py.

API usage

A single question can trigger several OpenAI requests. SQL generation, SQL review, and business-summary generation typically require separate model calls. Additional calls may occur when SQL requires revision or an ambiguous forecast horizon needs interpretation.

Forecast calculation, historical backtesting, and forecast-chart construction run locally and do not make additional OpenAI requests. The application currently uses gpt-4o-mini for its agent roles to limit cost.

Security

Keep API keys in .env only.

Do not publish production or customer databases.

If an API key was committed previously, revoke it and create a replacement.

If .env was committed at any point, adding it to .gitignore later is not enough. Remove it from tracking with git rm --cached .env, commit the removal, and rotate every credential it contained.

