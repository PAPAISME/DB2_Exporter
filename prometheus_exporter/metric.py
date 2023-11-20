"""Helpers around prometheus_client to create and register metrics."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Type

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Enum,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)
from prometheus_client.metrics import MetricWrapperBase


@dataclass(frozen=True)
class MetricType:
    """Details about a metric type."""

    cls: Type[MetricWrapperBase]
    options: List[str] = field(default_factory=list)


# Map metric types to their MetricTypes
METRIC_TYPES: dict[str, MetricType] = {
    "counter": MetricType(cls=Counter),
    "gauge": MetricType(cls=Gauge),
    "histogram": MetricType(cls=Histogram, options=["buckets"]),
    "summary": MetricType(cls=Summary),
    "enum": MetricType(cls=Enum, options=["states"]),
    "info": MetricType(cls=Info),
}


@dataclass
class MetricConfig:
    """Metric configuration."""

    name: str
    description: str
    type: str
    labels: Iterable[str] = field(default_factory=tuple)
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = tuple(sorted(self.labels))

        if self.type not in METRIC_TYPES:
            raise InvalidMetricType(self.name, self.type)


class MetricsRegistry:
    """Registry for metrics."""

    registry: CollectorRegistry

    def __init__(self):
        self.registry = CollectorRegistry(auto_describe=True)
        self._metrics: dict[str, MetricWrapperBase] = {}

    def create_metrics(
        self, configs: Iterable[MetricConfig]
    ) -> dict[str, MetricWrapperBase]:
        """Create metrics from a list of MetricConfigs."""

        metrics: Dict[str, MetricWrapperBase] = {
            config.name: self._produce_metric_instance(config) for config in configs
        }

        self._metrics.update(metrics)

        return metrics

    def get_metric(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> MetricWrapperBase:
        """Return a metric, optionally configured with labels."""

        metric = self._metrics[name]

        if labels:
            return metric.labels(**labels)

        return metric

    def get_metrics(self) -> Dict[str, MetricWrapperBase]:
        """Return a dict mapping names to metrics."""

        return self._metrics.copy()

    def generate_latest_metrics(self) -> bytes:
        """Generate text with the lastest version metrics values from the registry."""

        return bytes(generate_latest(self.registry))

    def _produce_metric_instance(self, config: MetricConfig) -> MetricWrapperBase:
        """By each metric type to produce the corresponding metric instance."""

        metric_type = METRIC_TYPES[config.type]

        options = {
            key: value
            for key, value in config.options.items()
            if key in metric_type.options
        }

        return metric_type.cls(
            name=config.name,
            documentation=config.description,
            labelnames=config.labels,
            registry=self.registry,
            **options,
        )


class InvalidMetricType(Exception):
    """Raised when invalid metric type is found."""

    def __init__(self, name: str, invalid_type: str):
        self.name = name
        self.invalid_type = invalid_type

        type_list = ", ".join(sorted(METRIC_TYPES))

        super().__init__(f"Invalid type for {self.name}: must be one of {type_list}")
