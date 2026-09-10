"""Centralized environment-driven configuration for DataSage."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        case_sensitive=False,
    )

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    database_file: Path = Field(
        validation_alias=AliasChoices("DB_FILE", "DATASAGE_DB_FILE")
    )
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    request_timeout_seconds: float = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")
    sql_row_limit: int = Field(default=100, alias="SQL_ROW_LIMIT")
    sqlite_timeout_seconds: float = Field(default=10, alias="SQLITE_TIMEOUT_SECONDS")
    chart_sample_rows: int = Field(default=75, alias="CHART_SAMPLE_ROWS")
    summary_sample_rows: int = Field(default=15, alias="SUMMARY_SAMPLE_ROWS")
    max_sql_retries: int = Field(default=2, alias="MAX_SQL_RETRIES")
    max_forecast_retries: int = Field(default=1, alias="MAX_FORECAST_RETRIES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_llm_columns: str = Field(default="", alias="ALLOWED_LLM_COLUMNS")
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    auth_required: bool = Field(default=False, alias="AUTH_REQUIRED")
    app_password: SecretStr = Field(
        default=SecretStr(""), alias="DATASAGE_APP_PASSWORD"
    )

    @field_validator("database_file")
    @classmethod
    def resolve_database(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def llm_column_allowlist(self) -> set[str]:
        return {item.strip() for item in self.allowed_llm_columns.split(",") if item.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.database_file.is_file():
        raise ValueError("The configured database file does not exist or is inaccessible.")
    return settings
