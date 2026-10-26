"""Configuration: YAML file, CLI overrides and secrets from the environment.

Secrets (NCBI key, contact email) are read from `.env` / the process environment only.
They are never read from YAML, never serialised and never logged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

CONFIG_ENV_VAR = "SYNERGYRX_CONFIG"
DEFAULT_CONFIG_NAME = "config.yaml"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ConfigError(ValueError):
    """Raised when the configuration is missing or inconsistent."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HostLimit(_Model):
    rate: float = Field(1.0, gt=0, description="Requests per second.")
    burst: int = Field(1, ge=1, description="Token bucket capacity.")
    max_concurrency: int = Field(1, ge=1, description="Simultaneous requests to the host.")


class RetrySettings(_Model):
    max_attempts: int = Field(5, ge=1)
    backoff_base: float = Field(1.0, gt=0)
    backoff_max: float = Field(60.0, gt=0)
    max_retry_after: float = Field(300.0, gt=0, description="Give up if a server asks for more.")


class CircuitSettings(_Model):
    failure_threshold: int = Field(5, ge=1)
    reset_seconds: float = Field(300.0, gt=0)


class NcbiSettings(_Model):
    rate_without_key: float = 3.0
    rate_with_key: float = 10.0
    hosts: list[str] = Field(default_factory=lambda: ["eutils.ncbi.nlm.nih.gov"])


class HttpSettings(_Model):
    product: str = "synergyrx-data"
    project_url: str | None = None
    require_contact_email: bool = True
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    max_response_bytes: int = 64 * 1024**2
    max_download_bytes: int = 4 * 1024**3
    sample_max_bytes: int = 64 * 1024**2
    cache_ttl_seconds: int = 86_400
    honor_crawl_delay: bool = True
    robots_for_bulk: bool = True
    safety_factor: float = Field(0.9, gt=0, le=1, description="Run this far below each limit.")
    default_limit: HostLimit = Field(default_factory=HostLimit)
    hosts: dict[str, HostLimit] = Field(default_factory=dict)
    ncbi: NcbiSettings = Field(default_factory=NcbiSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit: CircuitSettings = Field(default_factory=CircuitSettings)


class SourceConfig(_Model):
    enabled: bool = True
    version: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class CleaningSettings(_Model):
    manual_overrides: str = "config/manual_drug_overrides.csv"
    cell_line_aliases: str = "config/cell_line_aliases.csv"
    review_confidence_below: float = 0.7
    max_synonyms: int = 50
    score_outlier_abs: float = Field(200.0, description="Scores beyond this are flagged.")
    conflict_abs_tolerance: float = Field(10.0, description="Range above this is a conflict.")
    expression_top_variable_genes: int | None = None


class LiteratureSettings(_Model):
    min_synonym_length: int = 4
    max_ngram: int = 6
    min_drugs_per_sentence: int = 2
    annotation_sample_size: int = Field(300, ge=200, le=500)
    annotation_max_per_pmid: int = 3
    annotation_seed: int = 20260101


class ExportSettings(_Model):
    mock_seed: int = 42
    mock_examples: int = 200
    max_synonyms: int = 10


class LoggingSettings(_Model):
    level: str = "INFO"
    json_logs: bool = True
    file: str | None = None
    progress: bool = True


class Secrets(BaseModel):
    """Values that come from `.env` only."""

    model_config = ConfigDict(extra="ignore")

    ncbi_api_key: SecretStr | None = None
    contact_email: str | None = None
    ncbi_tool: str = "synergyrx-data"

    @field_validator("contact_email")
    @classmethod
    def _valid_email(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not _EMAIL_RE.match(value):
            raise ValueError("CONTACT_EMAIL is not a valid email address")
        return value

    @property
    def api_key(self) -> str | None:
        return self.ncbi_api_key.get_secret_value() if self.ncbi_api_key else None

    def values_to_redact(self) -> list[str]:
        return [v for v in (self.api_key, self.contact_email) if v]


class Settings(_Model):
    project_root: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Path("data")
    # Generated files are grouped by format under `data_dir`: `data/parquet` (the processed
    # tables), `data/csv`, `data/json` and `data/reports` (Markdown / HTML documents).
    # Each can be pointed elsewhere; left unset it follows `data_dir`.
    parquet_dir: Path | None = None
    csv_dir: Path | None = None
    json_dir: Path | None = None
    reports_dir: Path | None = None
    manifest_path: Path = Path("manifest.json")
    licenses_path: Path = Path("LICENSES.md")
    sources_doc_path: Path = Path("docs/SOURCES.md")

    sample: int | None = Field(None, ge=1)
    dry_run: bool = False
    refresh: bool = False

    http: HttpSettings = Field(default_factory=HttpSettings)
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
    literature: LiteratureSettings = Field(default_factory=LiteratureSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True, repr=False)

    def resolve(self, path: Path | str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def relative(self, path: Path) -> str:
        """Project-relative POSIX path, used as a stable key in the manifest."""
        try:
            return path.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    @property
    def data_root(self) -> Path:
        return self.resolve(self.data_dir)

    def _output(self, override: Path | None, default: str) -> Path:
        return self.resolve(override) if override else self.data_root / default

    @property
    def parquet_root(self) -> Path:
        """The processed tables: the final layer after `raw` and `interim`."""
        return self._output(self.parquet_dir, "parquet")

    @property
    def csv_root(self) -> Path:
        """Every CSV output: coverage, conflicts, review queues, the annotation sample."""
        return self._output(self.csv_dir, "csv")

    @property
    def json_root(self) -> Path:
        """JSON for the frontend and the JSON Lines copy of the annotation sample."""
        return self._output(self.json_dir, "json")

    @property
    def reports_root(self) -> Path:
        """Documents meant to be read: the quality report and the annotation guide."""
        return self._output(self.reports_dir, "reports")

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def manual_dir(self) -> Path:
        return self.data_root / "manual"

    @property
    def cache_dir(self) -> Path:
        return self.data_root / "cache" / "http"

    @property
    def state_dir(self) -> Path:
        return self.data_root / "state"


def _split_override_key(key: str) -> list[str]:
    """Split `a.b[c.d].e` into ['a', 'b', 'c.d', 'e'] (brackets protect dots in host names)."""
    parts: list[str] = []
    for bracketed, plain in re.findall(r"\[([^\]]+)\]|([^.\[\]]+)", key):
        parts.append(bracketed or plain)
    if not parts:
        raise ConfigError(f"empty override key: {key!r}")
    return parts


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `dotted.key=value` overrides; values are parsed as YAML scalars."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value, got {item!r}")
        key, _, value = item.partition("=")
        node = raw
        *parents, leaf = _split_override_key(key.strip())
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"cannot override {key!r}: {part!r} is not a mapping")
            node = child
        node[leaf] = yaml.safe_load(value)
    return raw


def load_secrets(project_root: Path, environ: dict[str, str] | None = None) -> Secrets:
    """Read secrets from `<project_root>/.env`, with real environment variables winning."""
    values: dict[str, str | None] = {}
    env_file = project_root / ".env"
    if env_file.is_file():
        values.update(dotenv_values(env_file))
    values.update(os.environ if environ is None else environ)
    return Secrets(
        ncbi_api_key=SecretStr(key)
        if (key := (values.get("NCBI_API_KEY") or "").strip())
        else None,
        contact_email=values.get("CONTACT_EMAIL") or None,
        ncbi_tool=(values.get("NCBI_TOOL") or "synergyrx-data").strip(),
    )


def find_config(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    if env_path := os.environ.get(CONFIG_ENV_VAR):
        return find_config(Path(env_path))
    candidate = Path.cwd() / DEFAULT_CONFIG_NAME
    return candidate if candidate.is_file() else None


def load_settings(
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    *,
    project_root: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Build settings from YAML (optional), then overrides, then secrets from the environment."""
    path = find_config(config_path)
    raw: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a YAML mapping")
        raw = loaded
    if "secrets" in raw:
        raise ConfigError("secrets must live in .env, not in the YAML configuration")
    raw = apply_overrides(raw, overrides or [])
    root = project_root or (path.parent if path is not None else Path.cwd())
    raw.setdefault("project_root", str(raw.pop("project_root", root)))
    try:
        settings = Settings.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc
    settings.secrets = load_secrets(settings.project_root, environ)
    return settings
