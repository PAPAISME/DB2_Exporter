"""Configuration management functions."""

import os
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from logging import Logger
from typing import IO, Any, Iterable, Union, FrozenSet

import jsonschema
import yaml
from dotenv import load_dotenv

from prometheus_exporter.metric import MetricConfig

from src.db import (
    DATABASE_LABEL,
    DataBaseError,
    InvalidQueryParameters,
    InvalidQuerySchedule,
    Query,
    QueryMetric,
    create_db_engine,
)

# Metric for counting database errors
DB_ERRORS_METRIC_NAME = "database_errors"
_DB_ERRORS_METRIC_CONFIG = MetricConfig(
    name=DB_ERRORS_METRIC_NAME,
    description="Number of database errors",
    type="counter",
)

# Metric for counting performed queries
QUERIES_METRIC_NAME = "queries"
_QUERIES_METRIC_CONFIG = MetricConfig(
    name=QUERIES_METRIC_NAME,
    description="Number of database queries",
    type="counter",
    labels=("query", "status"),
)
# Metric for counting queries execution latency
QUERY_LATENCY_METRIC_NAME = "query_latency"
_QUERY_LATENCY_METRIC_CONFIG = MetricConfig(
    name=QUERY_LATENCY_METRIC_NAME,
    description="Query execution latency",
    type="histogram",
    labels=("query",),
)

GLOBAL_METRICS = frozenset(
    [DB_ERRORS_METRIC_NAME, QUERIES_METRIC_NAME, QUERY_LATENCY_METRIC_NAME]
)

# Regexp for validating environment variables names
_ENV_VAR_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*$")

_MULTIPLIERS = {"s": 1, "m": 60, "h": 3600, "d": 3600 * 24}


class ConfigError(Exception):
    """Configuration is invalid."""


@dataclass(frozen=True)
class DataBaseConfig:
    """Configuration for a database."""

    name: str
    dsn: str
    labels: dict[str, str] = field(default_factory=dict)
    keep_connected: bool = True

    def __post_init__(self):
        try:
            create_db_engine(self.dsn)
        except DataBaseError as error:
            raise ConfigError(str(error)) from error


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    databases: dict[str, DataBaseConfig]
    metrics: dict[str, MetricConfig]
    queries: dict[str, Query]


# Type matching os.environ.
Environ = Mapping[str, str]

# Content for the "parameters" config option
ParametersConfig = list[dict[str, Any]]


def load_config(config_fd: IO, logger: Logger) -> Config:
    """Load YAML config from file."""

    load_dotenv()

    data = defaultdict(dict, yaml.safe_load(config_fd))

    _validate_config(data)

    databases, database_labels = _get_databases(data["databases"])

    extra_labels = frozenset([DATABASE_LABEL]) | database_labels

    metrics = _get_metrics(data["metrics"], extra_labels)

    queries = _get_queries(data["queries"], frozenset(databases), metrics)

    config = Config(databases, metrics, queries)

    _warn_if_unused(config, logger)

    return config


def _get_databases(
    database_configs: dict[str, dict[str, Any]]
) -> tuple[dict[str, DataBaseConfig], FrozenSet[str]]:
    """Return a dict mapping names to database configs,
    and a set of database labels."""

    databases_info = {}
    all_db_labels: set[FrozenSet[str]] = set()  # set of all labels sets

    try:
        for name, database_config in database_configs.items():
            labels = database_config.get("labels", {})
            all_db_labels.add(frozenset(labels))

            databases_info[name] = DataBaseConfig(
                name=name,
                dsn=str(os.getenv(database_config["dsn"])),
                labels=labels,
                keep_connected=database_config.get("keep_connected", True),
            )
    except Exception as error:
        raise ConfigError(str(error)) from error

    db_labels: FrozenSet[str]

    if not all_db_labels:
        db_labels = frozenset()
    elif len(all_db_labels) > 1:
        raise ConfigError("Not all databases define the same labels")
    else:
        db_labels = all_db_labels.pop()

    return databases_info, db_labels


def _get_metrics(
    metric_configs: dict[str, dict[str, Any]], extra_labels: FrozenSet[str]
) -> dict[str, MetricConfig]:
    """Return a dict mapping metric names to their configuration."""

    metrics_info = {}

    # Global metrics
    for global_metric in (
        _DB_ERRORS_METRIC_CONFIG,
        _QUERIES_METRIC_CONFIG,
        _QUERY_LATENCY_METRIC_CONFIG,
    ):
        metrics_info[global_metric.name] = MetricConfig(
            name=global_metric.name,
            description=global_metric.description,
            type=global_metric.type,
            labels=set(global_metric.labels) | extra_labels,
            options=global_metric.options,
        )

    # Configed metrics
    for name, metric_config in metric_configs.items():
        _validate_metric_config(name, metric_config, extra_labels)

        description = metric_config.pop("description", "")
        metric_type = metric_config.pop("type")
        labels = set(metric_config.pop("labels", ())) | extra_labels
        metric_config["expiration"] = _convert_interval(metric_config.get("expiration"))

        metrics_info[name] = MetricConfig(
            name=name,
            description=description,
            type=metric_type,
            labels=labels,
            options=metric_config,
        )

    return metrics_info


def _validate_metric_config(
    name: str, config: dict[str, Any], extra_labels: FrozenSet[str]
):
    """Validate a metric configuration."""

    if name in GLOBAL_METRICS:
        raise ConfigError(f"Label name '{name}' is reserved for builtin metric")

    labels = set(config.get("labels", ()))
    overlap_labels = labels & extra_labels

    if overlap_labels:
        overlap_list = ", ".join(sorted(overlap_labels))

        raise ConfigError(
            f"Labels for metric '{name}' overlap with reserved/database ones: {overlap_list}"
        )


def _get_queries(
    query_configs: dict[str, dict[str, Any]],
    database_names: FrozenSet[str],
    metrics: dict[str, MetricConfig],
) -> dict[str, Query]:
    """Return a list of Queries from config."""

    metric_names = frozenset(metrics)
    queries_info: dict[str, Query] = {}

    for name, config in query_configs.items():
        include_databases: FrozenSet[str]

        if isinstance(config["include_databases"], list):
            include_databases = frozenset(config["include_databases"])
        else:
            include_databases = database_names

        exclude_databases = frozenset(config["exclude_databases"])

        _validate_query_config(
            name=name,
            config=config,
            database_names=database_names,
            include_databases=include_databases,
            exclude_databases=exclude_databases,
            metric_names=metric_names,
        )

        query_metrics = _get_query_metrics(config, metrics)
        parameters = config.get("parameters")

        query_args = {
            "include_databases": list(include_databases),
            "exclude_databases": list(exclude_databases),
            "metrics": query_metrics,
            "sql": config["sql"].strip(),
            "timeout": config.get("timeout"),
            "interval": _convert_interval(config.get("interval")),
            "schedule": config.get("schedule"),
        }

        try:
            if parameters:
                queries_info.update(
                    (
                        f"{name}[params{index}]",
                        Query(
                            name=f"{name}[params{index}]",
                            parameters=params,
                            **query_args,
                        ),
                    )
                    for index, params in enumerate(parameters)
                )
            else:
                queries_info[name] = Query(name, **query_args)
        except (InvalidQueryParameters, InvalidQuerySchedule) as error:
            raise ConfigError(str(error)) from error

    return queries_info


def _get_query_metrics(
    config: dict[str, Any],
    metrics: dict[str, MetricConfig],
) -> list[QueryMetric]:
    """Return QueryMetrics for a query."""

    return [
        QueryMetric(name=name, labels=sorted(set(metrics["name"].labels)))
        for name in config["metrics"]
    ]


def _validate_query_config(
    name: str,
    config: dict[str, Any],
    database_names: FrozenSet[str],
    include_databases: FrozenSet[str],
    exclude_databases: FrozenSet[str],
    metric_names: FrozenSet[str],
):
    """Validate a query configuration."""

    unknown_databases = include_databases - exclude_databases - database_names

    if unknown_databases:
        unknown_list = ", ".join(sorted(unknown_databases))

        raise ConfigError(f'Unknown databases for query "{name}": {unknown_list}')

    unknown_metrics = set(config["metrics"]) - metric_names

    if unknown_metrics:
        unknown_list = ", ".join(sorted(unknown_metrics))

        raise ConfigError(f'Unknown metrics for query "{name}": {unknown_list}')

    parameters = config.get("parameters")

    if parameters:
        keys = {frozenset(param.keys()) for param in parameters}

        if len(keys) > 1:
            raise ConfigError(
                f"Invalid parameters definition for query '{name}': "
                "parameters dictionaries must all have the same keys"
            )


def _convert_interval(interval: Union[int, str, None]) -> Union[int, None]:
    """Convert a time interval to seconds.

    Return None if no interval is specified.
    """

    if interval is None:
        return None

    multiplier = 1

    if isinstance(interval, str):
        suffix = interval[-1]

        if suffix in _MULTIPLIERS:
            interval = interval[:-1]
            multiplier = _MULTIPLIERS[suffix]

    return int(interval) * multiplier


def _validate_config(config: dict[str, Any]):
    schema_file = resources.files("src") / "schemas" / "config.yaml"
    schema = yaml.safe_load(schema_file.read_bytes())

    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as error:
        path = "/".join(str(item) for item in error.absolute_path)

        raise ConfigError(f"Invalid config at {path}: {error.message}") from error


def _warn_if_unused(config: Config, logger: Logger):
    """Warn if there are unused databases or metrics defined."""

    used_dbs: set[str] = set()
    used_metrics: set[str] = set()

    for query in config.queries.values():
        used_dbs.update(
            list(set(query.include_databases) - set(query.exclude_databases))
        )

        used_metrics.update(metric.name for metric in query.metrics)

    unused_dbs = sorted(set(config.databases) - used_dbs)

    if unused_dbs:
        logger.warning(
            f"unused entries in \"databases\" section: {', '.join(unused_dbs)}"
        )

    unused_metrics = sorted(set(config.metrics) - GLOBAL_METRICS - used_metrics)

    if unused_metrics:
        logger.warning(
            f"unused entries in \"metrics\" section: {', '.join(unused_metrics)}"
        )
