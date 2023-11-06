"""Run a web server to provide a Prometheus metrics endpoint."""

from contextlib import asynccontextmanager
from textwrap import dedent
from typing import Any, Dict, List

from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.metrics import MetricWrapperBase

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from Master_Python.log_config import log_config
from Master_Python.metric import MetricConfig, MetricsRegistry


class PrometheusExporter:
    """Use a web server to expose metrics to Prometheus."""

    name = "db2-prometheus-exporter"
    description = ""
    host = "localhost"
    port = 8000
    registry: MetricsRegistry
    metrics_path = "/metrics"
    app: FastAPI

    def __init__(self) -> None:
        self.registry = MetricsRegistry()
        self.app = FastAPI(lifespan=self.lifespan)

        @self.app.get("/", response_class=HTMLResponse)
        def home() -> Any:
            """Home page request handler."""

            if self.description:
                title = f"{self.name} - {self.description}"
            else:
                title = self.name

            text = dedent(
                f"""<!DOCTYPE html>
                <html>
                  <head>
                    <title>{title}</title>
                    <meta name="generator" content="{self.name}">
                  </head>
                  <body>
                    <h1>{title}</h1>
                    <p>
                      Metric are exported at the
                      <a href=".{self.metrics_path}">{self.metrics_path}</a>
                      endpoint.
                    </p>
                  </body>
                </html>
                """
            )

            return text

        @self.app.get(self.metrics_path)
        async def update_metrics() -> Any:
            """Update metrics."""

            await self.update_handler(self.registry.get_metrics())

            response = Response(
                content=self.registry.generate_latest_metrics(),
                status_code=200,
                media_type=CONTENT_TYPE_LATEST,
            )

            return response

    def configure(self):
        """Childclass can implement this function.

        This function is called to load configuration and then create metrics.
        """

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> None:
        """Childclass can implement this function.

        FastAPI use the lifespan to define the startup and shutdown logic.

        Args:
            app (FastAPI): FastAPI Instance
        """

        yield

    async def update_handler(self, metrics: Dict[str, MetricWrapperBase]) -> None:
        """Childclass can implement this function."""

    def create_metrics(
        self, metric_configs: List[MetricConfig]
    ) -> Dict[str, MetricWrapperBase]:
        """Create and register metrics from MetricConfigs."""

        return self.registry.create_metrics(metric_configs)

    def run(self) -> None:
        """Run uvicorn command to startup the exporter web server."""

        self.configure()

        uvicorn.run(
            app=self.app,
            host=self.host,
            port=self.port,
            reload=False,
            use_colors=False,
            log_config=log_config,
        )
