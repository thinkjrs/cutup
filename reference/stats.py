"""Computes reference portfolio statistics for cutup's integration tests.

    uv run --project reference python reference/stats.py           # write
    uv run --project reference python reference/stats.py --check   # compare

Reads tests/fixtures/datasets.json and writes tests/fixtures/stats.json, which
tests/reference.rs checks Returns::stats against. For each dataset, on equal weights, on
long-short weights that sum to 1 and on weights that don't, each statistic comes from its
closed form on NumPy's sample mean and covariance (np.cov, ddof=1), independent of cutup's
computation from the portfolio's return in each period:

* expected return μᵀw and volatility sqrt(wᵀΣw);
* the Sharpe ratio (μᵀw − r) / sqrt(wᵀΣw) at r = 0 and r = 0.0004;
* the diversification ratio σᵀw / sqrt(wᵀΣw);
* each asset's share of the variance wᵢ(Σw)ᵢ / wᵀΣw;
* CVaR at β = 0.5, 0.8 and 0.95: the average loss over the worst 1 − β of the periods, the
  last one in part, written out from that definition.

Each case is cross-checked against PyPortfolioOpt's portfolio_performance, to 1e-12, and
against riskfolio-lib's Risk_Contribution with rm="MV", which uses central differences, to
1e-6, and its CVaR_Hist, to 1e-12. A portfolio whose volatility is below 1e-10 of the
sizes of its positions' largest returns, Σᵢ|wᵢ| maxₜ|rₜᵢ|, counts as riskless, as cutup counts it: its ratios
and shares are null.

The with_covariance cases are the same weights under three covariances, for
Returns::stats_with: a shrinkage of 0.2 toward the average variance, the exponentially weighted
covariance at a half-life of 5 (pandas' ewm(halflife=5).cov(), as reference/covariance.py
computes it), and a given matrix, (S + diag(S)) / 2, written into the case. The volatility,
Sharpe ratios, diversification ratio and risk shares come from the closed forms on that matrix,
cross-checked against PyPortfolioOpt's portfolio_performance to 1e-12 and riskfolio-lib's
Risk_Contribution to 1e-6. The expected return and CVaR don't depend on the covariance, so the
Rust test checks they're those of Returns::stats. A variance within 10nε of the square of the
positions' own volatilities, Σᵢ|wᵢ|σᵢ under the covariance, counts as riskless, as cutup counts it.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if any value moved by more than 1e-12 of its size or any case changed shape.
"""

import json
import sys

import numpy as np
import pandas as pd
import riskfolio
from pypfopt.base_optimizer import portfolio_performance

from covariance import exponentially_weighted
from fixtures import FIXTURES, write_json

DATASETS = FIXTURES / "datasets.json"
OUTPUT = FIXTURES / "stats.json"
RATES = [0.0, 0.0004]
BETAS = [0.5, 0.8, 0.95]


def load_returns(dataset: dict) -> np.ndarray:
    data = pd.DataFrame(dataset["data"], dtype=float)
    if dataset["kind"] == "prices":
        data = data.pct_change().dropna()
    return data.to_numpy()


def cvar(portfolio: np.ndarray, beta: float) -> float:
    """The average loss over the worst 1 − β of the periods, the last one in part."""
    losses = np.sort(-portfolio)[::-1]
    tail = (1.0 - beta) * len(losses)
    whole = int(np.floor(tail))
    total = losses[:whole].sum() + (tail - whole) * (losses[whole] if whole < len(losses) else 0.0)
    return float(total / tail)


def stats(returns: np.ndarray, weights: np.ndarray) -> dict:
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False, ddof=1)
    portfolio = returns @ weights
    expected = float(mu @ weights)
    variance = float(weights @ cov @ weights)
    volatility = np.sqrt(max(variance, 0.0))
    # cutup's rule: below 1e-10 of the sizes of the positions' largest returns.
    sizes = np.abs(weights) @ np.abs(returns).max(axis=0)
    riskless = volatility <= 1e-10 * sizes
    out = {
        "expected_return": expected,
        "volatility": 0.0 if riskless else float(volatility),
        "sharpe_ratios": [None if riskless else (expected - r) / volatility for r in RATES],
        "diversification_ratio": None if riskless else float(np.sqrt(np.diag(cov)) @ weights / volatility),
        "risk_contributions": None if riskless else (weights * (cov @ weights) / variance).tolist(),
        "cvar": [cvar(portfolio, beta) for beta in BETAS],
    }
    cross_check(returns, mu, cov, weights, out)
    return out


def close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-300)


def cross_check(returns, mu, cov, weights, out) -> None:
    """Checks the closed forms against PyPortfolioOpt and riskfolio-lib."""
    for rate, sharpe in zip(RATES, out["sharpe_ratios"]):
        # Its Sharpe ratio divides by a volatility of 0 on a riskless portfolio, unused here.
        with np.errstate(divide="ignore", invalid="ignore"):
            expected, volatility, theirs = portfolio_performance(weights, mu, cov, risk_free_rate=rate)
        assert close(expected, out["expected_return"], 1e-12), (expected, out)
        if sharpe is not None:
            assert close(volatility, out["volatility"], 1e-12), (volatility, out)
            assert close(theirs, sharpe, 1e-12), (theirs, sharpe)
    if out["risk_contributions"] is not None:
        frame = pd.DataFrame(returns)
        contributions = riskfolio.Risk_Contribution(
            pd.DataFrame(weights), frame, cov=pd.DataFrame(cov), rm="MV"
        )
        shares = np.asarray(contributions, dtype=float).ravel() / out["volatility"]
        assert np.allclose(shares, out["risk_contributions"], rtol=0, atol=1e-6), (shares, out)
    portfolio = returns @ weights
    for beta, value in zip(BETAS, out["cvar"]):
        theirs = riskfolio.CVaR_Hist(portfolio, alpha=1.0 - beta)
        assert close(theirs, value, 1e-12) or abs(theirs - value) <= 1e-15, (beta, theirs, value)


def covariance_stats(returns: np.ndarray, cov: np.ndarray, weights: np.ndarray) -> dict:
    """The statistics that depend on the covariance, from their closed forms under `cov`."""
    mu = returns.mean(axis=0)
    variance = float(weights @ cov @ weights)
    volatility = np.sqrt(max(variance, 0.0))
    sizes = np.abs(weights) @ np.sqrt(np.diag(cov))
    riskless = variance <= 10 * len(weights) * np.finfo(float).eps * sizes**2
    expected = float(mu @ weights)
    out = {
        "volatility": 0.0 if riskless else float(volatility),
        "sharpe_ratios": [None if riskless else (expected - r) / volatility for r in RATES],
        "diversification_ratio": None if riskless else float(np.sqrt(np.diag(cov)) @ weights / volatility),
        "risk_contributions": None if riskless else (weights * (cov @ weights) / variance).tolist(),
    }
    if not riskless:
        for rate, sharpe in zip(RATES, out["sharpe_ratios"]):
            _, theirs_volatility, theirs = portfolio_performance(weights, mu, cov, risk_free_rate=rate)
            assert close(theirs_volatility, out["volatility"], 1e-12), (theirs_volatility, out)
            assert close(theirs, sharpe, 1e-12), (theirs, sharpe)
        contributions = riskfolio.Risk_Contribution(
            pd.DataFrame(weights), pd.DataFrame(returns), cov=pd.DataFrame(cov), rm="MV"
        )
        shares = np.asarray(contributions, dtype=float).ravel() / out["volatility"]
        assert np.allclose(shares, out["risk_contributions"], rtol=0, atol=1e-6), (shares, out)
    return out


def covariances(returns: np.ndarray) -> list[tuple[dict, np.ndarray]]:
    """Each covariance config, as the Rust test builds it, and its matrix."""
    sample = np.cov(returns, rowvar=False, ddof=1)
    n = sample.shape[0]
    shrunk = 0.2 * np.trace(sample) / n * np.eye(n) + 0.8 * sample
    given = (sample + np.diag(np.diag(sample))) / 2
    return [
        ({"shrinkage": 0.2}, shrunk),
        ({"half_life": 5.0}, exponentially_weighted(returns, 5.0)),
        ({"matrix": given.tolist()}, given),
    ]


def covariance_cases() -> list[dict]:
    datasets = json.loads(DATASETS.read_text())
    out = []
    for name, dataset in datasets.items():
        returns = load_returns(dataset)
        for config, cov in covariances(returns):
            for weights in weight_sets(name, returns.shape[1]):
                out.append({
                    "dataset": name,
                    "covariance": config,
                    "weights": weights,
                    **covariance_stats(returns, cov, np.array(weights)),
                })
    return out


def weight_sets(name: str, n: int) -> list[list[float]]:
    rng = np.random.default_rng(sum(map(ord, name)))
    noise = rng.normal(scale=0.4, size=n)
    long_short = np.full(n, 1.0 / n) + noise - noise.mean()
    unnormalized = np.round(rng.uniform(-0.5, 1.0, size=n), 3)
    sets = [np.full(n, 1.0 / n), long_short, unnormalized]
    if name == "collinear2":
        # Asset 1 is 2 × asset 0 + 0.001: (2, −1) returns −0.001 every period.
        sets.append(np.array([2.0, -1.0]))
    return [[float(w) for w in weights] for weights in sets]


def cases() -> list[dict]:
    datasets = json.loads(DATASETS.read_text())
    out = []
    for name, dataset in datasets.items():
        returns = load_returns(dataset)
        for weights in weight_sets(name, returns.shape[1]):
            out.append({"dataset": name, "weights": weights, **stats(returns, np.array(weights))})
    return out


def values(case: dict) -> list:
    """Every number in a case, in order, with None for a missing ratio."""
    flat = []
    for key, value in case.items():
        if key == "dataset":
            continue
        if key == "covariance":
            for item in value.values():
                if isinstance(item, list):
                    for row in item:
                        flat.extend(row)
                else:
                    flat.append(item)
            continue
        if isinstance(value, list):
            flat.extend(value)
        else:
            flat.append(value)
    return flat


def drift(committed: list, fresh: list) -> list[str]:
    if len(committed) != len(fresh):
        return ["case count"]
    problems = []
    for i, (old, new) in enumerate(zip(committed, fresh)):
        a, b = values(old), values(new)
        if old["dataset"] != new["dataset"] or len(a) != len(b):
            problems.append(f"case {i} changed shape")
        elif any((x is None) != (y is None) or (x is not None and not close(x, y, 1e-12) and abs(x - y) > 1e-15) for x, y in zip(a, b)):
            problems.append(f"case {i} ({new['dataset']}) drifted")
    return problems


def main() -> None:
    fresh = {"rates": RATES, "betas": BETAS, "cases": cases(), "with_covariance": covariance_cases()}
    if "--check" in sys.argv[1:]:
        committed = json.loads(OUTPUT.read_text())
        problems = drift(committed["cases"], fresh["cases"])
        problems += [f"with_covariance {p}" for p in drift(committed["with_covariance"], fresh["with_covariance"])]
        if committed["rates"] != RATES or committed["betas"] != BETAS:
            problems.append("rates or betas")
        for problem in problems:
            print(problem)
        total = len(fresh["cases"]) + len(fresh["with_covariance"])
        print(f"{len(problems)} of {total} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)
    write_json(OUTPUT, fresh)
    print(f"wrote {len(fresh['cases'])} cases and {len(fresh['with_covariance'])} with a covariance to {OUTPUT}")


if __name__ == "__main__":
    main()
