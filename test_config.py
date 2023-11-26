"""Test for config.py"""

import logging

import pytest
import yaml

from src.config import (
    DB_ERRORS_METRIC_NAME,
    GLOBAL_METRICS,
    QUERIES_METRIC_NAME,
    QUERY_LATENCY_METRIC_NAME,
    ConfigError,
    load_config,
)
from src.db import QueryMetric


@pytest.fixture(name="logger")
def logger_fixture(caplog):
    """Logger."""

    with caplog.at_level("DEBUG"):
        yield logging.getLogger()

    caplog.clear()


@pytest.fixture(name="config_full")
def config_full_fixture():
    """Full Correct Config."""

    yield {
        "databases": {"db": {"dsn": "sqlite://"}},
        "metrics": {"m": {"type": "gauge", "labels": ["l1", "l2"]}},
        "queries": {
            "q": {
                "interval": 10,
                "include_databases": ["db"],
                "metrics": ["m"],
                "sql": "SELECT 1 as m",
            }
        },
    }


@pytest.fixture(name="write_config")
def write_config_fixture(tmp_path):
    """Write Config."""

    path = tmp_path / "config"

    def write(data):
        path.write_text(yaml.dump(data), "utf-8")

        return path

    yield write


CONFIG_UNKNOWN_DBS = {
    "databases": {},
    "metrics": {"m": {"type": "summary"}},
    "queries": {
        "q": {
            "interval": 10,
            "include_databases": ["db1", "db2"],
            "metrics": ["m"],
            "sql": "SELECT 1",
        }
    },
}

CONFIG_UNKNOWN_METRICS = {
    "databases": {"db": {"dsn": "sqlite://"}},
    "metrics": {},
    "queries": {
        "q": {
            "interval": 10,
            "include_databases": ["db"],
            "metrics": ["m1", "m2"],
            "sql": "SELECT 1",
        }
    },
}

CONFIG_MISSING_DB_KEY = {
    "databases": {},
    "metrics": {},
    "queries": {"q1": {"interval": 10}},
}

CONFIG_MISSING_METRIC_TYPE = {
    "databases": {"db": {"dsn": "sqlite://"}},
    "metrics": {"m": {}},
    "queries": {},
}

CONFIG_INVALID_METRIC_NAME = {
    "databases": {"db": {"dsn": "sqlite://"}},
    "metrics": {"is wrong": {"type": "gauge"}},
    "queries": {},
}

CONFIG_INVALID_LABEL_NAME = {
    "databases": {"db": {"dsn": "sqlite://"}},
    "metrics": {"m": {"type": "gauge", "labels": ["wrong-name"]}},
    "queries": {},
}

CONFIG_INVALID_METRICS_PARAMS_DIFFERENT_KEYS = {
    "databases": {"db": {"dsn": "sqlite://"}},
    "metrics": {"m": {"type": "gauge"}},
    "queries": {
        "q": {
            "interval": 10,
            "include_databases": ["db"],
            "metrics": ["m"],
            "sql": "SELECT :param AS m",
            "parameters": [{"foo": 1}, {"bar": 2}],
        },
    },
}


class TestLoadConfig:
    """Test for load config function."""

    def test_load_databases_section(self, logger, write_config):
        """Test for load database section.

        The 'databases' section is loaded from the config file.

        """

        config = {
            "databases": {
                "db1": {"dsn": "sqlite:///foo"},
                "db2": {
                    "dsn": "sqlite:///bar",
                    "keep_connected": False,
                },
            },
            "metrics": {},
            "queries": {},
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        assert {"db1", "db2"} == set(result.databases)

        database1 = result.databases["db1"]
        database2 = result.databases["db2"]

        assert database1.name == "db1"
        assert database1.dsn == "sqlite:///foo"
        assert database1.keep_connected

        assert database2.name == "db2"
        assert database2.dsn == "sqlite:///bar"
        assert not database2.keep_connected

    def test_load_databases_missing_dsn(self, logger, write_config):
        """Test for load databases missing dsn.

        An error is raised if the 'dsn' key is missing for a database.

        """

        config = {"databases": {"db1": {}}, "metrics": {}, "queries": {}}

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == "Invalid config at databases/db1: 'dsn' is a required property"
        )

    def test_load_databases_invalid_dsn(self, logger, write_config):
        """Test for load databases invalid dsn.

        An error is raised if the DSN is invalid.

        """

        config = {
            "databases": {"db1": {"dsn": "invalid_dsn"}},
            "metrics": {},
            "queries": {},
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert str(error.value) == "Invalid database DSN: 'invalid_dsn'"

    def test_load_databases_no_labels(self, logger, write_config):
        """Test for load databases no labels.

        If no labels are defined, an empty dict is returned.

        """

        config = {
            "databases": {
                "db": {
                    "dsn": "sqlite://",
                }
            },
            "metrics": {},
            "queries": {},
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        db = result.databases["db"]

        assert db.labels == {}

    def test_load_databases_labels(self, logger, write_config):
        """Test for load databases labels.

        Labels can be defined for databases.

        """

        config = {
            "databases": {
                "db": {
                    "dsn": "sqlite://",
                    "labels": {"label1": "value1", "label2": "value2"},
                }
            },
            "metrics": {},
            "queries": {},
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        db = result.databases["db"]

        assert db.labels == {"label1": "value1", "label2": "value2"}

    def test_load_databases_labels_not_all_same(self, logger, write_config):
        """Test for load databases labels not all same.

        If not all databases have the same labels, an error is raised.

        """

        config = {
            "databases": {
                "db1": {
                    "dsn": "sqlite://",
                    "labels": {"label1": "value1", "label2": "value2"},
                },
                "db2": {
                    "dsn": "sqlite://",
                    "labels": {"label2": "value2", "label3": "value3"},
                },
            },
            "metrics": {},
            "queries": {},
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert str(error.value) == "Not all databases define the same labels"

    def test_load_metrics_section(self, logger, write_config):
        """Test for load metrics section.

        The 'metrics' section is loaded from the config file.

        """

        config = {
            "databases": {"db1": {"dsn": "sqlite://"}},
            "metrics": {
                "metric1": {
                    "type": "summary",
                    "description": "metric one",
                    "labels": ["label1", "label2"],
                    "expiration": "2m",
                },
                "metric2": {
                    "type": "histogram",
                    "description": "metric two",
                    "buckets": [10, 100, 1000],
                },
                "metric3": {
                    "type": "enum",
                    "description": "metric three",
                    "states": ["on", "off"],
                    "expiration": 100,
                },
            },
            "queries": {},
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        metric1 = result.metrics["metric1"]

        assert metric1.type == "summary"
        assert metric1.description == "metric one"
        assert metric1.labels == ("database", "label1", "label2")
        assert metric1.options == {"expiration": 120}

        metric2 = result.metrics["metric2"]

        assert metric2.type == "histogram"
        assert metric2.description == "metric two"
        assert metric2.labels == ("database",)
        assert metric2.options == {
            "buckets": [10, 100, 1000],
            "expiration": None,
        }

        metric3 = result.metrics["metric3"]

        assert metric3.type == "enum"
        assert metric3.description == "metric three"
        assert metric3.labels == ("database",)
        assert metric3.options == {
            "states": ["on", "off"],
            "expiration": 100,
        }

        # Global metrics
        assert result.metrics.get(DB_ERRORS_METRIC_NAME) is not None
        assert result.metrics.get(QUERIES_METRIC_NAME) is not None
        assert result.metrics.get(QUERY_LATENCY_METRIC_NAME) is not None

    def test_load_metrics_overlap_reserved_label(self, logger, write_config):
        """Test for load metrics overlap reserved label.

        An error is raised if reserved labels are used.

        """

        config = {
            "databases": {"db1": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "gauge", "labels": ["database"]}},
            "queries": {},
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == "Labels for metric 'm' overlap with reserved/database ones: database"
        )

    def test_load_metrics_overlap_database_label(self, logger, write_config):
        """Test for load metrics overlap database label.

        An error is raised if database labels are used for metrics.

        """

        config = {
            "databases": {"db1": {"dsn": "sqlite://", "labels": {"l1": "v1"}}},
            "metrics": {"m": {"type": "gauge", "labels": ["l1"]}},
            "queries": {},
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == "Labels for metric 'm' overlap with reserved/database ones: l1"
        )

    @pytest.mark.parametrize("global_name", list(GLOBAL_METRICS))
    def test_load_metrics_reserved_name(
        self, logger, config_full, write_config, global_name
    ):
        """Test for load metrics reserved name.

        An error is raised if a reserved label name is used.

        """

        config_full["metrics"][global_name] = {"type": "counter"}

        config_file = write_config(config_full)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == f"Label name '{global_name}' is reserved for builtin metric"
        )

    def test_load_metrics_unsupported_type(self, logger, write_config):
        """Test for load metrics unsupported type.

        An error is raised if an unsupported metric type is passed.

        """

        config = {
            "databases": {"db1": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "info", "description": "info type metric"}},
            "queries": {},
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert str(error.value) == (
            "Invalid config at metrics/m/type: 'info' is not one of "
            "['counter', 'enum', 'gauge', 'histogram', 'summary']"
        )

    @pytest.mark.parametrize(
        "expiration,value",
        [
            (10, 10),
            ("10", 10),
            ("10s", 10),
            ("10m", 600),
            ("1h", 3600),
            ("1d", 3600 * 24),
            (None, None),
        ],
    )
    def test_load_metrics_expiration(
        self, logger, config_full, write_config, expiration, value
    ):
        """Test for load metrics expiration.

        The metric series expiration time can be specified with suffixes.

        """

        config_full["metrics"]["m"]["expiration"] = expiration

        config_file = write_config(config_full)

        with config_file.open() as fd:
            config = load_config(fd, logger)

        assert config.metrics["m"].options["expiration"] == value

    def test_load_queries_section(self, logger, write_config):
        """Test for load queires section.

        The 'queries' section is loaded from the config file.

        """

        config = {
            "databases": {
                "db1": {"dsn": "sqlite:///foo"},
                "db2": {"dsn": "sqlite:///bar"},
            },
            "metrics": {
                "m1": {"type": "summary", "labels": ["l1", "l2"]},
                "m2": {"type": "histogram"},
            },
            "queries": {
                "q1": {
                    "interval": 10,
                    "include_databases": ["db1"],
                    "metrics": ["m1"],
                    "sql": "SELECT 1",
                },
                "q2": {
                    "interval": 10,
                    "include_databases": ["db2"],
                    "metrics": ["m2"],
                    "sql": "SELECT 2",
                },
            },
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        query1 = result.queries["q1"]

        assert query1.name == "q1"
        assert query1.databases == ["db1"]
        assert query1.metrics == [QueryMetric("m1", ["database", "l1", "l2"])]
        assert query1.sql == "SELECT 1"
        assert query1.parameters == {}

        query2 = result.queries["q2"]

        assert query2.name == "q2"
        assert query2.databases == ["db2"]
        assert query2.metrics == [QueryMetric("m2", ["database"])]
        assert query2.sql == "SELECT 2"
        assert query2.parameters == {}

    def test_load_queries_section_with_parameters(self, logger, write_config):
        """Test for load queries section with parameters.

        Queries can have parameters.

        """

        config = {
            "databases": {"db": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "summary", "labels": ["l"]}},
            "queries": {
                "q": {
                    "interval": 10,
                    "include_databases": ["db"],
                    "metrics": ["m"],
                    "sql": "SELECT :param1 AS l, :param2 AS m",
                    "parameters": [
                        {"param1": "label1", "param2": 10},
                        {"param1": "label2", "param2": 20},
                    ],
                },
            },
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        query1 = result.queries["q[params0]"]

        assert query1.name == "q[params0]"
        assert query1.databases == ["db"]
        assert query1.metrics == [QueryMetric("m", ["database", "l"])]
        assert query1.sql == "SELECT :param1 AS l, :param2 AS m"
        assert query1.parameters == {
            "param1": "label1",
            "param2": 10,
        }

        query2 = result.queries["q[params1]"]

        assert query2.name == "q[params1]"
        assert query2.databases == ["db"]
        assert query2.metrics == [QueryMetric("m", ["database", "l"])]
        assert query2.sql == "SELECT :param1 AS l, :param2 AS m"
        assert query2.parameters == {
            "param1": "label2",
            "param2": 20,
        }

    def test_load_queries_section_with_wrong_parameters(self, logger, write_config):
        """Test for load queries section with wrong parameters.

        An error is raised if query parameters don't match.

        """

        config = {
            "databases": {"db": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "summary", "labels": ["l"]}},
            "queries": {
                "q": {
                    "interval": 10,
                    "include_databases": ["db"],
                    "metrics": ["m"],
                    "sql": "SELECT :param1 AS l, :param3 AS m",
                    "parameters": [
                        {"param1": "label1", "param2": 10},
                        {"param1": "label2", "param2": 20},
                    ],
                },
            },
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == "Parameters for query 'q[params0]' don't match those from SQL"
        )

    def test_load_queries_section_include_databases_with_string_type(
        self, logger, write_config
    ):
        """Test for load queries section include databases with string type.

        Include databases with string type

        """

        config = {
            "databases": {
                "db1": {"dsn": "sqlite:///foo"},
                "db2": {"dsn": "sqlite:///bar"},
            },
            "metrics": {
                "m": {"type": "summary", "labels": ["l1", "l2"]},
            },
            "queries": {
                "q": {
                    "interval": 10,
                    "include_databases": "All",
                    "metrics": ["m"],
                    "sql": "SELECT 1",
                },
            },
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        query = result.queries["q"]

        assert (query.databases) == ["db1", "db2"]

    def test_load_queries_section_with_exclude_databases(self, logger, write_config):
        """Test for load queries section with exclude_databases.

        The item in exclude_databases will be remove from include_databases.

        """

        config = {
            "databases": {
                "db1": {"dsn": "sqlite:///foo"},
                "db2": {"dsn": "sqlite:///bar"},
            },
            "metrics": {
                "m": {"type": "summary", "labels": ["l1", "l2"]},
            },
            "queries": {
                "q": {
                    "interval": 10,
                    "include_databases": "All",
                    "exclude_databases": ["db1"],
                    "metrics": ["m"],
                    "sql": "SELECT 1",
                },
            },
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        query = result.queries["q"]

        assert (query.databases) == ["db2"]

    def test_load_queries_section_with_schedule_and_interval(
        self, logger, write_config
    ):
        """Test for load queries section with schedule and interval.

        An error is raised if query schedule and interval are both present.

        """

        config = {
            "databases": {"db": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "summary"}},
            "queries": {
                "q": {
                    "include_databases": ["db"],
                    "metrics": ["m"],
                    "sql": "SELECT 1",
                    "interval": 10,
                    "schedule": "0 * * * *",
                },
            },
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == "Invalid schedule for query 'q': both interval and schedule specified"
        )

    def test_load_queries_section_invalid_schedule(self, logger, write_config):
        """Test for queries section invalid schedule.

        An error is raised if query schedule has wrong format.

        """

        config = {
            "databases": {"db": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "summary"}},
            "queries": {
                "q": {
                    "include_databases": ["db"],
                    "metrics": ["m"],
                    "sql": "SELECT 1",
                    "schedule": "wrong",
                },
            },
        }

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == "Invalid schedule for query 'q': invalid schedule format"
        )

    def test_load_queries_section_timeout(self, logger, config_full, write_config):
        """Test for load queries section timeout.

        Query configuration can include a timeout.

        """

        config_full["queries"]["q"]["timeout"] = 2.0

        config_file = write_config(config_full)

        with config_file.open() as fd:
            result = load_config(fd, logger)

        query1 = result.queries["q"]

        assert query1.timeout == 2.0

    @pytest.mark.parametrize(
        "timeout,error_message",
        [
            (
                -1.0,
                "Invalid config at queries/q/timeout: -1.0 is less than or equal to the minimum of 0",
            ),
            (
                0,
                "Invalid config at queries/q/timeout: 0 is less than or equal to the minimum of 0",
            ),
            (
                0.02,
                "Invalid config at queries/q/timeout: 0.02 is not a multiple of 0.1",
            ),
        ],
    )
    def test_load_queries_section_invalid_timeout(
        self, logger, config_full, write_config, timeout, error_message
    ):
        """Test for load queries section invalid timeout.

        An error is raised if query timeout is invalid.

        """

        config_full["queries"]["q"]["timeout"] = timeout

        config_file = write_config(config_full)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert str(error.value) == error_message

    @pytest.mark.parametrize(
        "config,error_message",
        [
            (CONFIG_UNKNOWN_DBS, "Unknown databases for query 'q': db1, db2"),
            (CONFIG_UNKNOWN_METRICS, "Unknown metrics for query 'q': m1, m2"),
            (
                CONFIG_MISSING_DB_KEY,
                "Invalid config at queries/q1: 'include_databases' is a required property",
            ),
            (
                CONFIG_MISSING_METRIC_TYPE,
                "Invalid config at metrics/m: 'type' is a required property",
            ),
            (
                CONFIG_INVALID_METRIC_NAME,
                "Invalid config at metrics: 'is wrong' does not match any "
                "of the regexes: '^[a-zA-Z_:][a-zA-Z0-9_:]*$'",
            ),
            (
                CONFIG_INVALID_LABEL_NAME,
                "Invalid config at metrics/m/labels/0: 'wrong-name' does not "
                "match '^[a-zA-Z_][a-zA-Z0-9_]*$'",
            ),
            (
                CONFIG_INVALID_METRICS_PARAMS_DIFFERENT_KEYS,
                "Invalid parameters definition for query 'q': "
                "parameters dictionaries must all have the same keys",
            ),
        ],
    )
    def test_configuration_incorrect(self, logger, write_config, config, error_message):
        """Test for configuration incorrect.

        An error is raised if configuration is incorrect.

        """

        config_file = write_config(config)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert str(error.value) == error_message

    def test_configuration_warning_unused(
        self, caplog, logger, config_full, write_config
    ):
        """Test for configuration warning unused.

        A warning is logged if unused entries are present in config.

        """

        config_full["databases"]["db2"] = {"dsn": "sqlite://"}
        config_full["databases"]["db3"] = {"dsn": "sqlite://"}
        config_full["metrics"]["m2"] = {"type": "gauge"}
        config_full["metrics"]["m3"] = {"type": "gauge"}

        config_file = write_config(config_full)

        with config_file.open() as fd:
            load_config(fd, logger)

        assert caplog.messages == [
            "unused entries in 'databases' section: db2, db3",
            "unused entries in 'metrics' section: m2, m3",
        ]

    def test_load_queries_missing_interval_default_to_none(self, logger, write_config):
        """Test for load queries missing interval default to none.

        If the interval is not specified, it defaults to None.

        """

        config = {
            "databases": {"db": {"dsn": "sqlite://"}},
            "metrics": {"m": {"type": "summary"}},
            "queries": {
                "q": {"include_databases": ["db"], "metrics": ["m"], "sql": "SELECT 1"}
            },
        }

        config_file = write_config(config)

        with config_file.open() as fd:
            config = load_config(fd, logger)

        assert config.queries["q"].interval is None

    @pytest.mark.parametrize(
        "interval,value",
        [
            (10, 10),
            ("10", 10),
            ("10s", 10),
            ("10m", 600),
            ("1h", 3600),
            ("1d", 3600 * 24),
            (None, None),
        ],
    )
    def test_load_queries_interval(
        self, logger, config_full, write_config, interval, value
    ):
        """Test for load queries interval.

        The query interval can be specified with suffixes.

        """

        config_full["queries"]["q"]["interval"] = interval

        config_file = write_config(config_full)

        with config_file.open() as fd:
            config = load_config(fd, logger)

        assert config.queries["q"].interval == value

    def test_load_queries_interval_not_specified(
        self, logger, config_full, write_config
    ):
        """Test for load queries interval not specified.

        If the interval is not specified, it's set to None.

        """

        del config_full["queries"]["q"]["interval"]

        config_file = write_config(config_full)

        with config_file.open() as fd:
            config = load_config(fd, logger)

        assert config.queries["q"].interval is None

    @pytest.mark.parametrize("interval", ["1x", "wrong", "1.5m"])
    def test_load_queries_invalid_interval_string(
        self, logger, config_full, write_config, interval
    ):
        """Test for load queries invalid interval string.

        An invalid string query interval raises an error.

        """

        config_full["queries"]["q"]["interval"] = interval

        config_file = write_config(config_full)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert str(error.value) == (
            "Invalid config at queries/q/interval: "
            f"'{interval}' does not match '^[0-9]+[smhd]?$'"
        )

    @pytest.mark.parametrize("interval", [0, -20])
    def test_load_queries_invalid_interval_number(
        self, logger, config_full, write_config, interval
    ):
        """Test for load queries invalid interval number.

        An invalid integer query interval raises an error.

        """

        config_full["queries"]["q"]["interval"] = interval

        config_file = write_config(config_full)

        with pytest.raises(ConfigError) as error, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(error.value)
            == f"Invalid config at queries/q/interval: {interval} is less than the minimum of 1"
        )

    def test_load_queries_no_metrics(self, logger, config_full, write_config):
        """Test for load queries no metrics.

        An error is raised if no metrics are configured.

        """

        config_full["queries"]["q"]["metrics"] = []

        config_file = write_config(config_full)

        with pytest.raises(ConfigError) as err, config_file.open() as fd:
            load_config(fd, logger)

        assert str(err.value) == "Invalid config at queries/q/metrics: [] is too short"

    def test_load_queries_no_databases(self, logger, config_full, write_config):
        """Test for load queries no databases.

        An error is raised if no databases are configured.

        """

        config_full["queries"]["q"]["include_databases"] = []

        config_file = write_config(config_full)

        with pytest.raises(ConfigError) as err, config_file.open() as fd:
            load_config(fd, logger)

        assert (
            str(err.value)
            == "Invalid config at queries/q/include_databases: [] is too short"
        )
