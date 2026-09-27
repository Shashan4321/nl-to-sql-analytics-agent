from pathlib import Path

import pytest

from nl2sql import warehouse
from nl2sql.db import Warehouse


@pytest.fixture(scope="session")
def wh(tmp_path_factory) -> Warehouse:
    path = Path(tmp_path_factory.mktemp("wh")) / "warehouse.duckdb"
    warehouse.build(path)
    return Warehouse(path)
