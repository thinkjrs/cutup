"""Shared paths and JSON formatting for the reference fixtures."""

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _format(value, indent: int) -> str:
    # Lists of numbers (a weight vector or a matrix row) go on one line so the
    # fixtures stay readable and diff cleanly; everything else is indented.
    pad = " " * indent
    if isinstance(value, dict):
        items = [
            f'{pad}  {json.dumps(key)}: {_format(item, indent + 2)}'
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
        rows = [f"{pad}  {_format(row, indent + 2)}" for row in value]
        return "[\n" + ",\n".join(rows) + f"\n{pad}]"
    if isinstance(value, list) and any(isinstance(v, dict) for v in value):
        items = [f"{pad}  {_format(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format(value, 0) + "\n")
