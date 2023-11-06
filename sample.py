"""Sample code to demo."""

import random
from contextlib import asynccontextmanager
from typing import Dict, cast

from prometheus_client import Counter, Gauge
from prometheus_client.metrics import MetricWrapperBase

from fastapi import FastAPI

from Master_Python.exporter import PrometheusExporter
from Master_Python.metric import MetricConfig


class SampleExporter(PrometheusExporter):
    """A sample exporter."""

    def __init__(self) -> None:
        super().__init__()

        self.name = "prometheus-sample-exporter"
        self.port = 9091

    def configure(self) -> None:
        self.create_metrics(
            [
                MetricConfig(
                    "a_gauge", "test gauge type", "gauge", labels=("foo", "bar")
                ),
                MetricConfig(
                    "a_counter", "test counter type", "counter", labels=("baz",)
                ),
            ]
        )

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> None:
        """FastAPI use the lifespan to define the startup and shutdown logic.

        Args:
            app (FastAPI): FastAPI Instance
        """

        yield

    async def update_handler(self, metrics: Dict[str, MetricWrapperBase]) -> None:
        """Test"""
        gauge = cast(Gauge, metrics["a_gauge"])

        gauge.labels(
            foo=random.choice(["this-foo", "other-foo"]),
            bar=random.choice(["this-bar", "other-bar"]),
        ).set(random.uniform(0, 100))

        counter = cast(Counter, metrics["a_counter"])

        counter.labels(
            baz=random.choice(["this-baz", "other-baz"]),
        ).inc(random.choice(range(10)))


sample_exporter = SampleExporter()

if __name__ == "__main__":
    sample_exporter.run()
