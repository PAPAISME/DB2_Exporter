"""Test"""

import itertools
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
import importlib_resources as resources
from logging import Logger
from pathlib import Path
from typing import IO, Any, Dict, List, Set, Union
from urllib.parse import quote_plus, urlencode

from dotenv import load_dotenv

import jsonschema
import yaml

from prometheus_exporter import metric


class ConfigError(Exception):
    """Configuration is invalid."""


if __name__ == "__main__":
    with open(
        "/Users/papaisme/Python_Practice/DB2_Exporter/src/schemas/oracle-stats.yaml",
        mode="r",
        encoding="utf-8",
    ) as config_fd:
        data = defaultdict(dict, yaml.safe_load(config_fd))

        print(data)

        schema_file = resources.files("src") / "schemas" / "config.yaml"

        print("\n\n\n")
        print(schema_file)

        schema = yaml.safe_load(schema_file.read_bytes())

        print("\n\n\n")
        print(schema)

        print("\n")
        print(type(schema))

        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as error:
            path = "/".join(str(item) for item in error.absolute_path)

            raise ConfigError(f"Invalid config at {path}: {error.message}") from error

        all_db_labels: set[frozenset[str]] = set()

        print(bool(all_db_labels))

        # all_db_labels.add(frozenset({"use": "opop1", "use1": "opop2"}))

        print(all_db_labels)

        all_db_labels.add(frozenset({}))

        print(bool(all_db_labels))

        print(all_db_labels)

        print(
            f"""Wrong column names from query: 
            expected Test, got Test1"""
        )

        load_dotenv()

        print(os.getenv("Test"))

        test_dict = {"a": 1, "b": 2}

        parameters = {3, 4, 5}
        
        confuse = [(f"[param{index}]", param) for index, param in enumerate(parameters)]
        
        print(confuse)

        test_dict.update(
            (f"[param{index}]", param) for index, param in enumerate(parameters)
        )

        print(test_dict)
