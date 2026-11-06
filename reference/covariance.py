"""Computes reference covariance estimates for cutup's tests.

    uv run --project reference python reference/covariance.py           # write
    uv run --project reference python reference/covariance.py --check   # compare

Reads tests/fixtures/datasets.json and writes tests/fixtures/covariance.json, which the
tests in src/covariance.rs check CovarianceConfig's estimates against: Ledoit-Wolf shrinkage,
the exponentially weighted covariance and the semicovariances. Its Black-Litterman cases are
checked by the tests in src/black_litterman.rs.

For each dataset and each Ledoit-Wolf target, the case holds the shrinkage intensity δ and the
shrunk covariance δF + (1 − δ)S, where S is the sample covariance with ddof=1, as cutup
computes it, and F the target built from S:

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

The exponentially weighted covariance, at half-lives of 1, 5 and 20 periods, is written out
from its definition: weights wₜ = λ^(T−1−t) with λ = 2^(−1/h), the weighted mean
m = Σwₜrₜ / Σwₜ, and Σwₜ(rₜ − m)(rₜ − m)ᵀ / (Σwₜ − Σwₜ² / Σwₜ). It's checked against pandas'
ewm(halflife=h).cov(), and riskfolio-lib's covar_matrix(method="ewma1", d=λ), to 1e-12.

The semicovariance below each asset's mean, Σₜ min(rₜ − r̄, 0) min(rₜ − r̄, 0)ᵀ / (T − 1), is
checked against riskfolio-lib's covar_matrix(method="semi"), and the one below a threshold of
0, Σₜ min(rₜ, 0) min(rₜ, 0)ᵀ / T, against PyPortfolioOpt's semicovariance(benchmark=0), to
1e-12.

The Black-Litterman posterior covariance, Σ + τΣ − τΣPᵀ(τPΣPᵀ + Ω)⁻¹PτΣ, is written out
from He and Litterman (1999) on four datasets with equal market weights: with default views
and τ = 1/T, checked against PyPortfolioOpt's BlackLittermanModel.bl_cov and riskfolio-lib's
black_litterman, which fixes τ at 1/T and uses the default Ω; with an explicit variance, a
confidence and a certain view at τ = 0.05, against PyPortfolioOpt's; and with no views, which
gives (1 + τ)Σ. All to 1e-12, except riskfolio-lib's: it inverts τΣ, so it's checked to 1e-15
times Σ's condition number when that's looser, and not on scales4, whose condition number of
about 1e16 leaves nothing to check. A view's default variance is τpᵀΣp, and a confidence c's
τ(1 − c)/c · pᵀΣp.

With one asset, or a sample covariance already equal to the target, there's nothing to
shrink and δ is 0, as scikit-learn has it. Where the constant correlation target is the
sample covariance but for rounding, as it always is for two assets, δ is made of that
rounding, so it's null and the covariance is the sample one.

With --check, nothing is written: the cases are recomputed and compared with the committed
file, failing if an intensity moved by more than 1e-12 of its size, a covariance entry by
more than 1e-12 of the covariance's largest, or any case changed shape.

Covariance entries are compared to the largest one because entries near 0 next to it, such
as the exact zeros of orthogonal3, are rounding that differs between BLAS builds.
"""

import json
import sys

import numpy as np
import pandas as pd
import riskfolio
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.risk_models import CovarianceShrinkage, semicovariance
from sklearn.covariance import ledoit_wolf, ledoit_wolf_shrinkage

from fixtures import FIXTURES, write_json

DATASETS = FIXTURES / "datasets.json"
OUTPUT = FIXTURES / "covariance.json"
TARGETS = ["constant_variance", "constant_correlation"]
HALF_LIVES = [1.0, 5.0, 20.0]
BL_DATASETS = ["two_blocks5", "prices10x60", "near_singular4", "scales4"]
DELTA = 2.5


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


def same_matrix(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-12) -> bool:
    """Equal to `tolerance` of the larger matrix's largest entry."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.abs(a - b).max() <= tolerance * max(np.abs(a).max(), np.abs(b).max())


def exponentially_weighted(returns: np.ndarray, half_life: float) -> np.ndarray:
    t = returns.shape[0]
    decay = 2.0 ** (-1.0 / half_life)
    weights = decay ** np.arange(t - 1, -1, -1, dtype=float)
    mean = weights @ returns / weights.sum()
    x = returns - mean
    cov = (x * weights[:, None]).T @ x / (weights.sum() - (weights**2).sum() / weights.sum())
    theirs = pd.DataFrame(returns).ewm(halflife=half_life).cov().iloc[-returns.shape[1]:].to_numpy()
    assert same_matrix(cov, theirs), (half_life, cov, theirs)
    frame = pd.DataFrame(returns)
    theirs = riskfolio.ParamsEstimation.covar_matrix(frame, method="ewma1", d=decay).to_numpy()
    assert same_matrix(cov, theirs), (half_life, cov, theirs)
    return cov


def semicovariances(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = returns.shape[0]
    below_mean = np.minimum(returns - returns.mean(axis=0), 0.0)
    mean = below_mean.T @ below_mean / (t - 1)
    theirs = riskfolio.ParamsEstimation.covar_matrix(pd.DataFrame(returns), method="semi").to_numpy()
    assert same_matrix(mean, theirs), (mean, theirs)
    below_zero = np.minimum(returns, 0.0)
    zero = below_zero.T @ below_zero / t
    theirs = semicovariance(pd.DataFrame(returns), returns_data=True, benchmark=0.0, frequency=1)
    assert same_matrix(zero, np.asarray(theirs)), (zero, theirs)
    return mean, zero


def view_sets(n: int, t: int) -> list[tuple[str, float, list[dict]]]:
    """Each view set's name, τ and views. Assets 0 to 3 exist in every dataset used."""
    basket = [0.0] * n
    basket[0], basket[1], basket[3] = 0.5, 0.5, -1.0
    default = [
        {"kind": "absolute", "assets": [0], "q": 0.002, "variance": None, "confidence": None},
        {"kind": "relative", "assets": [1, 2], "q": 0.001, "variance": None, "confidence": None},
    ]
    mixed = [
        {"kind": "absolute", "assets": [0], "q": 0.002, "variance": 1e-6, "confidence": None},
        {"kind": "relative", "assets": [1, 2], "q": 0.001, "variance": None, "confidence": 0.7},
        {"kind": "portfolio", "weights": basket, "q": 0.0005, "variance": 0.0, "confidence": None},
    ]
    return [("default", 1.0 / t, default), ("mixed", 0.05, mixed), ("none", 0.05, [])]


def view_portfolio(view: dict, n: int) -> np.ndarray:
    if view["kind"] == "portfolio":
        return np.array(view["weights"])
    p = np.zeros(n)
    p[view["assets"][0]] = 1.0
    if view["kind"] == "relative":
        p[view["assets"][1]] = -1.0
    return p


def black_litterman_cases() -> list[dict]:
    datasets = json.loads(DATASETS.read_text())
    out = []
    for name in BL_DATASETS:
        returns = load_returns(datasets[name])
        t, n = returns.shape
        cov = np.cov(returns, rowvar=False, ddof=1)
        market = np.full(n, 1.0 / n)
        for label, tau, views in view_sets(n, t):
            case = {"dataset": name, "views_name": label, "tau": tau, "views": views}
            if not views:
                case["covariance"] = ((1.0 + tau) * cov).tolist()
                out.append(case)
                continue
            p = np.array([view_portfolio(v, n) for v in views])
            q = np.array([v["q"] for v in views])
            spread = np.diag(p @ (tau * cov) @ p.T)
            omega = np.array([
                v["variance"] if v["variance"] is not None
                else spread[i] * ((1 - v["confidence"]) / v["confidence"] if v["confidence"] is not None else 1.0)
                for i, v in enumerate(views)
            ])
            tau_sigma_pt = tau * cov @ p.T
            a = p @ tau_sigma_pt + np.diag(omega)
            posterior = cov + tau * cov - tau_sigma_pt @ np.linalg.solve(a, tau_sigma_pt.T)
            pi = DELTA * cov @ market
            theirs = BlackLittermanModel(pd.DataFrame(cov), pi=pi, P=p, Q=q, omega=np.diag(omega), tau=tau).bl_cov()
            assert same_matrix(posterior, theirs.to_numpy()), (name, label)
            # riskfolio-lib inverts τΣ, so it's accurate only to rounding times Σ's condition
            # number, which for scales4 is about 1e16: too much to check anything.
            condition = np.linalg.cond(cov)
            if label == "default" and condition <= 1e10:
                _, theirs, _ = riskfolio.ParamsEstimation.black_litterman(
                    pd.DataFrame(returns), pd.DataFrame(market), p, q.reshape(-1, 1), delta=DELTA
                )
                tolerance = max(1e-12, 1e-15 * condition)
                assert same_matrix(posterior, theirs.to_numpy(), tolerance), (name, label)
            case["covariance"] = posterior.tolist()
            out.append(case)
    return out


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
        case["exponentially_weighted"] = [
            {"half_life": h, "covariance": exponentially_weighted(returns, h).tolist()}
            for h in HALF_LIVES
        ]
        mean, zero = semicovariances(returns)
        case["semicovariance"] = {"covariance": mean.tolist()}
        case["semicovariance_below"] = {"threshold": 0.0, "covariance": zero.tolist()}
        out.append(case)
    return out


def estimates(case: dict) -> list[tuple[str, dict]]:
    """Each estimate in a case, by name."""
    named = [(key, case[key]) for key in TARGETS + ["semicovariance", "semicovariance_below"]]
    for estimate in case["exponentially_weighted"]:
        named.append((f"exponentially_weighted({estimate['half_life']})", estimate))
    return named


def moved(old: dict, new: dict) -> str | None:
    """What moved between two cases' estimates, or None.

    An intensity or parameter is compared to 1e-12 of its size, and a covariance entry to
    1e-12 of the covariance's largest entry.
    """
    for key in sorted((old.keys() | new.keys()) - {"covariance"}):
        a, b = old.get(key), new.get(key)
        if (a is None) != (b is None):
            return key
        if a is not None and not (a == b or close(a, b, 1e-12)):
            return f"{key} {a} -> {b}"
    a, b = np.array(old["covariance"]), np.array(new["covariance"])
    if a.shape != b.shape:
        return "shape"
    gap = np.abs(a - b).max()
    if gap > 1e-12 * np.abs(a).max():
        return f"covariance by {gap:.3g} of {np.abs(a).max():.3g}"
    return None


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
            what = moved(a, b)
            if what is not None:
                problems.append(f"{new['dataset']} {name}: {what}")
    return problems


def black_litterman_drift(committed: list, fresh: list) -> list[str]:
    if len(committed) != len(fresh):
        return ["black_litterman case count"]
    problems = []
    for old, new in zip(committed, fresh):
        what = f"black_litterman/{new['dataset']}/{new['views_name']}"
        if {k: v for k, v in old.items() if k != "covariance"} != {k: v for k, v in new.items() if k != "covariance"}:
            problems.append(f"{what}: inputs changed")
        elif not same_matrix(old["covariance"], new["covariance"]):
            problems.append(f"{what}: covariance")
    return problems


def main() -> None:
    fresh = {"cases": cases(), "black_litterman": black_litterman_cases()}
    if "--check" in sys.argv[1:]:
        committed = json.loads(OUTPUT.read_text())
        problems = drift(committed["cases"], fresh["cases"])
        problems += black_litterman_drift(committed["black_litterman"], fresh["black_litterman"])
        for problem in problems:
            print(problem)
        drifted = len({problem.split()[0] for problem in problems})
        total = len(fresh["cases"]) + len(fresh["black_litterman"])
        print(f"{drifted} of {total} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)
    write_json(OUTPUT, fresh)
    print(f"wrote {len(fresh['cases'])} cases and {len(fresh['black_litterman'])} Black-Litterman cases to {OUTPUT}")


if __name__ == "__main__":
    main()
