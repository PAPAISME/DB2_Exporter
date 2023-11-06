"""Test for exporter.py"""

from unittest import mock

import pytest
from pytest_mock import MockerFixture

from fastapi.testclient import TestClient

from Master_Python.metric import MetricConfig
from Master_Python.exporter import PrometheusExporter

from Master_Python.log_config import log_config


@pytest.fixture(name="exporter")
def exporter_fixture() -> PrometheusExporter:
    """Exporter."""

    yield PrometheusExporter()


class TestPrometheusExporter:
    """Test for PrometheusExporter Class."""

    def test_create_metrics(self, exporter) -> None:
        """Test for create metrics."""

        configs = [
            MetricConfig("metric_name1", "desc1", "counter", {}),
            MetricConfig("metric_name2", "desc2", "histogram", {}),
        ]

        metrics = exporter.create_metrics(configs)

        assert len(metrics) == 2
        assert metrics["metric_name1"]._type == "counter"
        assert metrics["metric_name2"]._type == "histogram"

    def test_run_exporter(self, mocker: MockerFixture) -> None:
        """Test for run exporter."""

        mock_run = mocker.patch("Master_Python.exporter.uvicorn.run")

        mock_run.assert_called_with(
            mock.ANY,
            host="localhost",
            port=12345,
            reload=False,
            use_colors=False,
            log_config=log_config,
        )
