"""Computes reference expected-return estimates for cutup's tests.

    uv run --project reference python reference/expected_returns.py           # write
    uv run --project reference python reference/expected_returns.py --check   # compare

Reads tests/fixtures/datasets.json and writes tests/fixtures/expected_returns.json, which
tests/reference.rs checks Returns' expected-return estimates against.

The exponentially weighted mean, at half-lives of 1, 5 and 20 periods, is written out from
its definition: weights wₜ = λ^(T−1−t) with λ = 2^(−1/h), and m = Σwₜrₜ / Σwₜ. It's checked
against pandas' ewm(halflife=h).mean(), riskfolio-lib's mean_vector(method="ewma1", d=λ),
and PyPortfolioOpt's ema_historical_return with span = 2 / (1 − λ) − 1, per period and
without compounding, to 1e-12 of the largest return.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if a mean moved by more than 1e-12 of its dataset's largest return, or any
case changed shape.
"""

import json
import sys

import numpy as np
import pandas as pd
import riskfolio
from pypfopt.expected_returns import ema_historical_return

from covariance import load_returns
from fixtures import FIXTURES, write_json

DATASETS = FIXTURES / "datasets.json"
OUTPUT = FIXTURES / "expected_returns.json"
HALF_LIVES = [1.0, 5.0, 20.0]
TOLERANCE = 1e-12


def same(a, b, scale: float) -> bool:
    return np.abs(np.asarray(a) - np.asarray(b)).max() <= TOLERANCE * scale


def exponentially_weighted_mean(returns: np.ndarray, half_life: float) -> np.ndarray:
    t = returns.shape[0]
    decay = 2.0 ** (-1.0 / half_life)
    weights = decay ** np.arange(t - 1, -1, -1, dtype=float)
    mean = weights @ returns / weights.sum()

    scale = np.abs(returns).max()
    frame = pd.DataFrame(returns)
    theirs = frame.ewm(halflife=half_life).mean().iloc[-1].to_numpy()
    assert same(mean, theirs, scale), ("pandas", half_life)
    theirs = riskfolio.ParamsEstimation.mean_vector(frame, method="ewma1", d=decay).to_numpy()[0]
    assert same(mean, theirs, scale), ("riskfolio", half_life)
    span = 2.0 / (1.0 - decay) - 1.0
    theirs = ema_historical_return(
        frame, returns_data=True, compounding=False, span=span, frequency=1
    ).to_numpy()
    assert same(mean, theirs, scale), ("pypfopt", half_life)
    return mean


def cases() -> list[dict]:
    out = []
    for name, dataset in json.loads(DATASETS.read_text()).items():
        returns = load_returns(dataset)
        out.append(
            {
                "dataset": name,
                "scale": float(np.abs(returns).max()),
                "exponentially_weighted_mean": [
                    {"half_life": h, "mean": exponentially_weighted_mean(returns, h).tolist()}
                    for h in HALF_LIVES
                ],
            }
        )
    return out


def estimates(case: dict) -> list[tuple[str, list[float]]]:
    """Each estimate in a case, by name."""
    return [
        (f"exponentially_weighted_mean({e['half_life']})", e["mean"])
        for e in case["exponentially_weighted_mean"]
    ]


def drift(committed: list, fresh: list) -> list[str]:
    if len(committed) != len(fresh):
        return ["case count"]
    problems = []
    for old, new in zip(committed, fresh):
        old_estimates, new_estimates = estimates(old), estimates(new)
        if old["dataset"] != new["dataset"] or [k for k, _ in old_estimates] != [k for k, _ in new_estimates]:
            problems.append(f"{new['dataset']} changed shape")
            continue
        for (name, a), (_, b) in zip(old_estimates, new_estimates):
            if len(a) != len(b) or not same(a, b, new["scale"]):
                problems.append(f"{new['dataset']} {name}")
    return problems


def main() -> None:
    fresh = cases()
    if "--check" in sys.argv[1:]:
        committed = json.loads(OUTPUT.read_text())["cases"]
        problems = drift(committed, fresh)
        for problem in problems:
            print(problem)
        drifted = len({problem.split()[0] for problem in problems})
        print(f"{drifted} of {len(fresh)} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)
    write_json(OUTPUT, {"cases": fresh})
    print(f"wrote {len(fresh)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
