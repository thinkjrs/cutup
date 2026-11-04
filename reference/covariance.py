"""Computes reference Ledoit-Wolf covariances for cutup's tests.

    uv run --project reference python reference/covariance.py           # write
    uv run --project reference python reference/covariance.py --check   # compare

Reads tests/fixtures/datasets.json and writes tests/fixtures/covariance.json, which the
tests in src/covariance.rs check CovarianceConfig::ledoit_wolf against. For each dataset and
each target, the case holds the shrinkage intensity δ and the shrunk covariance
δF + (1 − δ)S, where S is the sample covariance with ddof=1, as cutup computes it, and F the
target built from S:

* constant_variance: F = μI, μ = trace(S) / n. δ is scikit-learn's ledoit_wolf_shrinkage,
  from Ledoit and Wolf, "A well-conditioned estimator for large-dimensional covariance
  matrices" (2004), on the 1/T moments of the centered returns.
* constant_correlation: F keeps each variance and puts the average correlation r̄ between
  every pair, F_ij = r̄·σ_i·σ_j. δ is from Ledoit and Wolf, "Honey, I shrunk the sample
  covariance matrix" (2004), written out from their covCor.m, on the same 1/T moments.

The intensities come from the 1/T moments the papers use, and the target scales with S, so
the shrunk covariance here is T/(T − 1) times the papers' 1/T one. The constant variance case is checked against scikit-learn's ledoit_wolf to 1e-12 on
that scale. PyPortfolioOpt's constant correlation estimate uses a ddof=1 sample covariance
with 1/T moments, which moves δ by O(1/T); given the 1/T covariance in its place, it's the
paper's estimate, and the constant correlation case is checked against it so, to 1e-12.

With one asset, or a sample covariance already equal to the target, there's nothing to
shrink and δ is 0, as scikit-learn has it. Where the constant correlation target is the
sample covariance but for rounding, as it always is for two assets, δ is made of that
rounding, so it's null and the covariance is the sample one.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if any value moved by more than 1e-12 of its size or any case changed shape.
"""

import json
import sys

import numpy as np
import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage
from sklearn.covariance import ledoit_wolf, ledoit_wolf_shrinkage

from fixtures import FIXTURES, write_json

DATASETS = FIXTURES / "datasets.json"
OUTPUT = FIXTURES / "covariance.json"
TARGETS = ["constant_variance", "constant_correlation"]


def load_returns(dataset: dict) -> np.ndarray:
    data = pd.DataFrame(dataset["data"], dtype=float)
    if dataset["kind"] == "prices":
        data = data.pct_change().dropna()
    return data.to_numpy()


def constant_correlation_target(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    sd = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(sd, sd)
    r_bar = (correlation.sum() - np.trace(correlation)) / (n * (n - 1))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(cov))
    return target


def constant_correlation_intensity(returns: np.ndarray) -> float:
    """δ from Ledoit and Wolf's covCor.m, on the 1/T sample covariance."""
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample)
    sd = np.sqrt(var)
    prior = constant_correlation_target(sample)
    y = x**2
    pi_mat = y.T @ y / t - sample**2
    pi_hat = pi_mat.sum()
    theta = (x**3).T @ x / t - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    r_bar = (prior[~np.eye(n, dtype=bool)] / np.outer(sd, sd)[~np.eye(n, dtype=bool)]).mean()
    rho_hat = np.trace(pi_mat) + r_bar * ((sd[None, :] / sd[:, None]) * theta).sum()
    gamma_hat = np.linalg.norm(sample - prior, "fro") ** 2
    return float(max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / t)))


def close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-300)


def cases() -> list[dict]:
    datasets = json.loads(DATASETS.read_text())
    out = []
    for name, dataset in datasets.items():
        returns = load_returns(dataset)
        t, n = returns.shape
        cov = np.cov(returns, rowvar=False, ddof=1)
        case = {"dataset": name}
        for target in TARGETS:
            if target == "constant_variance":
                intensity = float(ledoit_wolf_shrinkage(returns))
                prior = np.trace(cov) / n * np.eye(n)
                shrunk = intensity * prior + (1.0 - intensity) * cov
                theirs, delta = ledoit_wolf(returns)
                assert close(delta, intensity, 1e-12), (name, delta, intensity)
                assert np.allclose(shrunk * (t - 1) / t, theirs, rtol=0, atol=1e-12 * np.abs(theirs).max()), name
            else:
                prior = constant_correlation_target(cov)
                if np.linalg.norm(cov - prior) <= 1e-10 * np.linalg.norm(cov):
                    # The target is the sample covariance but for rounding, as it always is
                    # for two assets, so δ is made of rounding and can be anything.
                    intensity, shrunk = None, cov
                else:
                    intensity = constant_correlation_intensity(returns)
                    shrunk = intensity * prior + (1.0 - intensity) * cov
                    pypfopt = CovarianceShrinkage(pd.DataFrame(returns), returns_data=True, frequency=1)
                    pypfopt.S = np.cov(returns, rowvar=False, ddof=0)
                    theirs, delta = pypfopt._ledoit_wolf_constant_correlation()
                    assert close(delta, intensity, 1e-12), (name, delta, intensity)
                    assert np.allclose(shrunk * (t - 1) / t, theirs, rtol=0, atol=1e-12 * np.abs(theirs).max()), name
            case[target] = {"intensity": intensity, "covariance": shrunk.tolist()}
        out.append(case)
    return out


def values(case: dict) -> list:
    """Every number in a case, in order, with None for an intensity left out."""
    flat = []
    for target in TARGETS:
        flat.append(case[target]["intensity"])
        for row in case[target]["covariance"]:
            flat.extend(row)
    return flat


def drift(committed: list, fresh: list) -> list[str]:
    if len(committed) != len(fresh):
        return ["case count"]
    problems = []
    for old, new in zip(committed, fresh):
        a, b = values(old), values(new)
        if old["dataset"] != new["dataset"] or len(a) != len(b):
            problems.append(f"{new['dataset']} changed shape")
        elif any((x is None) != (y is None) or (x is not None and not close(x, y, 1e-12) and abs(x - y) > 1e-300) for x, y in zip(a, b)):
            problems.append(f"{new['dataset']} drifted")
    return problems


def main() -> None:
    fresh = {"cases": cases()}
    if "--check" in sys.argv[1:]:
        committed = json.loads(OUTPUT.read_text())
        problems = drift(committed["cases"], fresh["cases"])
        for problem in problems:
            print(problem)
        print(f"{len(problems)} of {len(fresh['cases'])} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)
    write_json(OUTPUT, fresh)
    print(f"wrote {len(fresh['cases'])} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
