"""Test for metric.py"""

from typing import (
    Any,
    Callable,
    cast,
)

from prometheus_client import Histogram
from prometheus_client.metrics import MetricWrapperBase
import pytest

from prometheus_exporter.metric import (
    InvalidMetricType,
    MetricConfig,
    MetricsRegistry,
)


class TestMetricConfig:
    """Test for MetricConfig class."""

    def test_invalid_metric_type(self) -> None:
        """Test invalid metric type."""

        with pytest.raises(InvalidMetricType) as error:
            MetricConfig("metric_name", "desc", "unknown")

        assert str(error.value) == (
            "Invalid type for metric_name: must be one of counter, enum, "
            "gauge, histogram, info, summary"
        )

    def test_labels_sorted(self) -> None:
        """Test labels sorted."""

        config = MetricConfig("metric_name", "desc", "counter", labels=("foo", "bar"))

        assert config.labels == ("bar", "foo")


class TestMetricsRegistry:
    """Test for MetricsRegistry class."""

    def test_create_metrics(self) -> None:
        """Test for create metrics."""

        configs = [
            MetricConfig("metric_name1", "desc1", "counter"),
            MetricConfig("metric_name2", "desc2", "histogram"),
        ]

        metrics = MetricsRegistry().create_metrics(configs)

        assert len(metrics) == 2
        assert metrics["metric_name1"]._type == "counter"
        assert metrics["metric_name2"]._type == "histogram"

    def test_create_metrics_with_options(self) -> None:
        """Test for create metrics with options."""

        configs = [
            MetricConfig(
                "metric_name", "desc", "histogram", options={"buckets": [10, 20]}
            )
        ]

        metrics = MetricsRegistry().create_metrics(configs)

        histogram = cast(Histogram, metrics["metric_name"])

        # Histogram has the two specified bucket plus +Inf
        assert len(histogram._buckets) == 3

    def test_create_metrics_options_ignores_unknown(self) -> None:
        """Test for create metrics with options ignore."""

        configs = [
            MetricConfig("metric_name", "desc", "gauge", options={"unknown": "value"})
        ]

        metrics = MetricsRegistry().create_metrics(configs)

        assert len(metrics) == 1

    def test_get_metric(self) -> None:
        """Test for get metric."""

        configs = [
            MetricConfig(
                "metric_name",
                "A test gauge",
                "gauge",
                labels=("l1", "l2"),
            )
        ]

        registry = MetricsRegistry()
        registry.create_metrics(configs)

        metric = registry.get_metric("metric_name")

        assert metric._name == "metric_name"
        assert metric._labelvalues == ()

    def test_get_metric_with_labels(self) -> None:
        """Test for get metric with labels."""

        configs = [
            MetricConfig("metric_name", "A test gauge", "gauge", labels=("l1", "l2"))
        ]

        registry = MetricsRegistry()
        registry.create_metrics(configs)

        metric = registry.get_metric("metric_name", {"l1": "v1", "l2": "v2"})

        assert metric._labelvalues == ("v1", "v2")

    def test_get_metrics(self) -> None:
        """Test for get metrics."""

        registry = MetricsRegistry()

        metrics = registry.create_metrics(
            [
                MetricConfig("metric_name1", "A test gauge", "gauge"),
                MetricConfig("metric_name2", "A test histogram", "histogram"),
            ]
        )

        assert registry.get_metrics() == metrics

    @pytest.mark.parametrize(
        "metric_type,options,action,text",
        [
            (
                "counter",
                {},
                lambda metric: metric.inc(),
                "counter\ntest_counter_total 1.0",
            ),
            ("gauge", {}, lambda metric: metric.set(12.3), "test_gauge 12.3"),
            (
                "histogram",
                {"buckets": [10, 20]},
                lambda metric: metric.observe(1.23),
                'test_histogram_bucket{le="10.0"} 1.0',
            ),
            (
                "summary",
                {},
                lambda metric: metric.observe(1.23),
                "test_summary_sum 1.23",
            ),
            (
                "enum",
                {"states": ["on", "off"]},
                lambda metric: metric.state("on"),
                'test_enum{test_enum="on"}',
            ),
            (
                "info",
                {},
                lambda metric: metric.info({"foo": "bar", "baz": "bza"}),
                'test_info_info{baz="bza",foo="bar"}',
            ),
        ],
    )
    def test_generate_latest_metrics(
        self,
        metric_type: str,
        options: dict[str, Any],
        action: Callable[[MetricWrapperBase], None],
        text: str,
    ) -> None:
        """Test for generate latest metrics."""

        registry = MetricsRegistry()

        name = "test_" + metric_type

        metrics = registry.create_metrics(
            [MetricConfig(name, "A test metric", metric_type, options=options)]
        )

        action(metrics[name])

        assert text in registry.generate_latest_metrics().decode("utf-8")
