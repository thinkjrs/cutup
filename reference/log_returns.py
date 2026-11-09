"""Computes reference log returns for cutup's tests.

    uv run --project reference python reference/log_returns.py           # write
    uv run --project reference python reference/log_returns.py --check   # compare

Reads tests/fixtures/datasets.json and writes tests/fixtures/log_returns.json, which the unit
tests in src/returns.rs check Returns::log_returns against.

Each dataset's log returns are NumPy's log1p of its simple returns, ln(1 + r), from pandas'
pct_change for the prices dataset, as cutup converts prices. They're checked against
PyPortfolioOpt's returns_from_prices(log_returns=True), given the prices, or for a dataset of
returns the prices that compound them from 1, and against ln(pₜ / pₜ₋₁), to 1e-12 of the
dataset's largest log return. PyPortfolioOpt takes np.log(1 + r), which loses the digits of a
return far below 1 that log1p keeps, so it's checked against the largest, not each entry.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if a log return moved by more than 4ε of itself, or a case changed shape.
"""

import json
import sys

import numpy as np
import pandas as pd
from pypfopt.expected_returns import returns_from_prices

from covariance import load_returns
from fixtures import FIXTURES, write_json

DATASETS = FIXTURES / "datasets.json"
OUTPUT = FIXTURES / "log_returns.json"
TOLERANCE = 1e-12
DRIFT = 4 * np.finfo(float).eps


def log_returns(dataset: dict) -> np.ndarray:
    simple = load_returns(dataset)
    logs = np.log1p(simple)

    if dataset["kind"] == "prices":
        prices = pd.DataFrame(dataset["data"], dtype=float)
    else:
        start = np.ones((1, logs.shape[1]))
        prices = pd.DataFrame(np.cumprod(np.vstack([start, 1.0 + simple]), axis=0))
    scale = np.abs(logs).max()
    theirs = returns_from_prices(prices, log_returns=True).to_numpy()
    gap = np.abs(logs - theirs).max()
    assert gap <= TOLERANCE * scale, ("pypfopt", gap)
    values = prices.to_numpy()
    gap = np.abs(logs - np.log(values[1:] / values[:-1])).max()
    assert gap <= TOLERANCE * scale, ("ln(p / p)", gap)
    return logs


def cases() -> list[dict]:
    return [
        {"dataset": name, "log_returns": log_returns(dataset).tolist()}
        for name, dataset in json.loads(DATASETS.read_text()).items()
    ]


def drift(committed: list, fresh: list) -> list[str]:
    if len(committed) != len(fresh):
        return ["case count"]
    problems = []
    for old, new in zip(committed, fresh):
        a, b = np.array(old["log_returns"]), np.array(new["log_returns"])
        if old["dataset"] != new["dataset"] or a.shape != b.shape:
            problems.append(f"{new['dataset']} changed shape")
        elif not (np.abs(a - b) <= DRIFT * np.abs(b)).all():
            problems.append(f"{new['dataset']} log returns")
    return problems


def main() -> None:
    fresh = cases()
    if "--check" in sys.argv[1:]:
        committed = json.loads(OUTPUT.read_text())["cases"]
        problems = drift(committed, fresh)
        for problem in problems:
            print(problem)
        print(f"{len(problems)} of {len(fresh)} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)
    write_json(OUTPUT, {"cases": fresh})
    print(f"wrote {len(fresh)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
