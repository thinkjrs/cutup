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

The CAPM returns, r_f + βᵢ(E[r_m] − r_f) with r_m the market portfolio's return each period
and βᵢ = Cov(rᵢ, r_m) / Var(r_m) from the ddof=1 sample covariance, are written out for equal
market weights at a risk-free rate of 0, and for weights n, n − 1, …, 1 at 1e-4. Each is
checked against PyPortfolioOpt's capm_return, per period and without compounding: with its
default market, the equal-weighted average, for the first, and given the market's returns for
the second, to 1e-12 of the largest return. A market with no variance, by cutup's rule for
Returns::stats, a volatility below 1e-10 of Σᵢ|wᵢ| maxₜ|rₜᵢ|, is expected to be an error.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if a mean moved by more than 1e-12 of its dataset's largest return, or any
case changed shape.
"""

import json
import sys

import numpy as np
import pandas as pd
import riskfolio
from pypfopt.expected_returns import capm_return, ema_historical_return

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


def capm(returns: np.ndarray, market: np.ndarray, rate: float) -> list[float] | None:
    """The CAPM returns against the market weights `market`, or None for a riskless market."""
    w = market / market.sum()
    m = returns @ w
    sizes = np.abs(w) @ np.abs(returns).max(axis=0)
    volatility = np.std(m, ddof=1)
    if volatility < 1e-10 * sizes:
        return None
    cov = np.cov(np.column_stack([returns, m]), rowvar=False)
    betas = cov[:-1, -1] / cov[-1, -1]
    mu = rate + betas * (m.mean() - rate)

    scale = np.abs(returns).max()
    frame = pd.DataFrame(returns)
    if np.all(market == market[0]):
        theirs = capm_return(frame, returns_data=True, risk_free_rate=rate, compounding=False, frequency=1)
    else:
        theirs = capm_return(
            frame,
            market_prices=pd.DataFrame({"mkt": m}),
            returns_data=True,
            risk_free_rate=rate,
            compounding=False,
            frequency=1,
        )
    assert same(mu, theirs.to_numpy(), scale), ("pypfopt capm", market, rate)
    return mu.tolist()


def capm_cases(n: int) -> list[tuple[list[float], float]]:
    """Equal market weights at a rate of 0, and weights n, n − 1, …, 1 at 1e-4."""
    return [([1.0] * n, 0.0), ([float(n - i) for i in range(n)], 1e-4)]


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
                "capm": [
                    {
                        "market_weights": market,
                        "risk_free_rate": rate,
                        "returns": capm(returns, np.array(market), rate),
                    }
                    for market, rate in capm_cases(returns.shape[1])
                ],
            }
        )
    return out


def estimates(case: dict) -> list[tuple[str, list[float] | None]]:
    """Each estimate in a case, by name, None for an expected error."""
    named = [
        (f"exponentially_weighted_mean({e['half_life']})", e["mean"])
        for e in case["exponentially_weighted_mean"]
    ]
    named += [
        (f"capm({e['market_weights']}, {e['risk_free_rate']})", e["returns"])
        for e in case["capm"]
    ]
    return named


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
            if (a is None) != (b is None) or (a is not None and (len(a) != len(b) or not same(a, b, new["scale"]))):
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
