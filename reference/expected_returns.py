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

The James-Stein mean, (1 − α)μ + α·μ̄1 with μ̄ the average of the sample means and
α = (tr Σ − 2λ_max) / (T‖μ − μ̄1‖²) clamped to [0, 1], is written out with Σ the ddof=1
sample covariance, and checked against riskfolio-lib's mean_vector(method="JS") where its
unclamped intensity is already in [0, 1], to 1e-12 of the largest return. With every mean
equal there's nothing to shrink, and it's the sample mean.

The Bayes-Stein mean (Jorion, 1986), (1 − α)μ + α·μ₀1 with μ₀ = 1ᵀΣ⁻¹μ / 1ᵀΣ⁻¹1 and
α = (n + 2) / (n + 2 + T(μ − μ₀1)ᵀΣ⁻¹(μ − μ₀1)), is written out in correlation units, as cutup
computes it, and checked against riskfolio-lib's mean_vector(method="BS", target="b2"), which
inverts Σ: to 1e-12 of the largest return, or 1e-15 times Σ's condition number where that's
looser, and not where that's above 1e10. A correlation matrix with an eigenvalue at most nε of
its largest is an expected error, singular to rounding, as cutup decides it. The case records
the correlation matrix's condition number, since the test's tolerance grows with it.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if a mean moved by more than 1e-12 of its dataset's largest return, or 1e-16
times the correlation matrix's condition number for Bayes-Stein where that's looser, or any
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


def same(a, b, scale: float, tolerance: float = TOLERANCE) -> bool:
    return np.abs(np.asarray(a) - np.asarray(b)).max() <= tolerance * scale


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
    if volatility <= 1e-10 * sizes:
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


def james_stein(returns: np.ndarray) -> list[float]:
    t, n = returns.shape
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    target = mu.mean()
    spread = ((mu - target) ** 2).sum()
    if spread == 0.0:
        return mu.tolist()
    raw = (np.trace(cov) - 2.0 * np.linalg.eigvalsh(cov).max()) / (t * spread)
    alpha = min(max(raw, 0.0), 1.0)
    shrunk = (1.0 - alpha) * mu + alpha * target
    if 0.0 <= raw <= 1.0:
        frame = pd.DataFrame(returns)
        theirs = riskfolio.ParamsEstimation.mean_vector(frame, method="JS", target="b1").to_numpy()[0]
        assert same(shrunk, theirs, np.abs(returns).max()), ("riskfolio JS", raw)
    return shrunk.tolist()


def correlation_condition(returns: np.ndarray) -> float | None:
    """The correlation matrix's condition number, which Bayes-Stein's tolerance grows with, or
    None where it isn't finite, which JSON can't hold; Bayes-Stein is an expected error there."""
    cov = np.cov(returns, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    condition = float(np.linalg.cond(cov / np.outer(sd, sd)))
    return condition if np.isfinite(condition) else None


def bayes_stein(returns: np.ndarray) -> list[float] | None:
    t, n = returns.shape
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    values, vectors = np.linalg.eigh(corr)
    if values.min() <= n * np.finfo(float).eps * values.max():
        return None

    def form(x, y):
        return ((vectors.T @ (x / sd)) * (vectors.T @ (y / sd)) / values).sum()

    ones = np.ones(n)
    target = form(ones, mu) / form(ones, ones)
    deviation = mu - target
    alpha = (n + 2) / (n + 2 + t * form(deviation, deviation))
    shrunk = (1.0 - alpha) * mu + alpha * target

    condition = np.linalg.cond(cov)
    if condition <= 1e10:
        frame = pd.DataFrame(returns)
        theirs = riskfolio.ParamsEstimation.mean_vector(frame, method="BS", target="b2").to_numpy()[0]
        scale = np.abs(returns).max()
        gap = np.abs(shrunk - theirs).max()
        assert gap <= max(1e-12, 1e-15 * condition) * scale, ("riskfolio BS", gap, condition)
    return shrunk.tolist()


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
                "james_stein": james_stein(returns),
                "bayes_stein": bayes_stein(returns),
                "bayes_stein_condition": correlation_condition(returns),
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
    named += [("james_stein", case["james_stein"]), ("bayes_stein", case["bayes_stein"])]
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
            # Σ⁻¹ amplifies rounding, which differs between BLAS builds, by the condition number.
            tolerance = TOLERANCE
            if name == "bayes_stein" and new["bayes_stein_condition"] is not None:
                tolerance = max(TOLERANCE, 1e-16 * new["bayes_stein_condition"])
            if (a is None) != (b is None) or (
                a is not None and (len(a) != len(b) or not same(a, b, new["scale"], tolerance))
            ):
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
