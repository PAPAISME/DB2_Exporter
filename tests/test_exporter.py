"""Test for exporter.py"""

import pytest
from pytest_mock import MockerFixture

from prometheus_exporter.metric import MetricConfig
from prometheus_exporter.exporter import PrometheusExporter

from prometheus_exporter.log_config import log_config


@pytest.fixture
def exporter_fixture() -> PrometheusExporter:
    """Exporter."""

    return PrometheusExporter()


class TestPrometheusExporter:
    """Test for PrometheusExporter Class."""

    def test_create_metrics(self, exporter: PrometheusExporter) -> None:
        """Test for create metrics."""

        configs = [
            MetricConfig("metric_name1", "desc1", "counter", {}),
            MetricConfig("metric_name2", "desc2", "histogram", {}),
        ]

        metrics = exporter.create_metrics(configs)

        assert len(metrics) == 2
        assert metrics["metric_name1"]._type == "counter"
        assert metrics["metric_name2"]._type == "histogram"

    def test_run_exporter(
        self, mocker: MockerFixture, exporter: PrometheusExporter
    ) -> None:
        """Test for run exporter."""

        mock_run = mocker.patch("Master_Python.exporter.uvicorn.run")

        exporter.run()

        mock_run.assert_called_with(
            app=exporter.app,
            host="localhost",
            port=8000,
            reload=False,
            use_colors=False,
            log_config=log_config,
        )
