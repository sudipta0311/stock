"""Write the current API OpenAPI document to the checked-in schema export."""

from __future__ import annotations

import json
from pathlib import Path

from wealth.api.main import app


def main() -> None:
    output_path = Path(__file__).resolve().parents[3] / "schemas" / "openapi.json"
    output_path.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
