"""Computes reference weights for cutup's integration tests.

    uv run --project reference python reference/generate.py           # write
    uv run --project reference python reference/generate.py --check   # compare

Reads tests/fixtures/datasets.json and writes tests/fixtures/expected.json, which
tests/reference.rs checks cutup against. Each method's reference is independent of
cutup's implementation:

* Equal weight: 1 / n.
* Inverse volatility and inverse variance: 1 / σ and 1 / σ² from np.var, normalized.
* Minimum variance and mean-variance: NumPy closed forms when the covariance is well
  conditioned, otherwise the minimum-norm solution of the KKT system from
  numpy.linalg.lstsq. Expected errors are decided from the null space of the
  covariance, not by running the same solver as cutup.
* Long-only minimum variance and mean-variance: support enumeration. On every subset of
  assets, the KKT system with the others held at zero gives that subset's optimum, if it
  is feasible. Among the candidates with no negative weights, the best objective is the
  long-only optimum, since each candidate is feasible and the optimum is one of them. This
  checks neither the multipliers' signs nor uses an iterative solver, unlike cutup's
  active-set method. Only covariances that aren't singular are used, where the optimum is
  unique. A target return outside the assets' expected returns is an error.
* Long-only max Sharpe: the same support enumeration, on the change of variables
  y = w / (μ − r_f)ᵀw that turns it into minimizing yᵀΣy subject to (μ − r_f)ᵀy = 1 and
  y ≥ 0. An unbounded Sharpe ratio is decided by scipy.optimize.linprog finding a
  long-only portfolio in Σ's null space (eigenvalues within n·ε of the largest, from
  numpy.linalg.eigh) with positive excess return.
* Max Sharpe: the NumPy closed form Σ⁻¹(μ − r_f) / 1ᵀΣ⁻¹(μ − r_f); the minimum-norm
  least-squares solution when Σ is singular.
* Risk parity: cyclical coordinate descent (Griveau-Billion et al., 2013) run to
  convergence, a different algorithm from cutup's Newton iteration. Expected errors are
  decided by finding a long-only portfolio with zero variance with scipy.optimize.nnls.
* HRP: PyPortfolioOpt's HRPOpt, for each linkage method.
* HERC: riskfolio-lib's HCPortfolio with model="HERC" and rm="MV", for each linkage
  method and for fixed numbers of clusters.

With --check, nothing is written: the cases are recomputed and compared with the
committed file, failing if any weight moved by more than a tenth of its tolerance or
any case changed shape. CI runs this so the fixtures can't drift from the pinned
reference implementations. The comparison is numeric rather than byte-for-byte
because NumPy's linear algebra can differ in the last bits between platforms.
"""

import importlib.metadata
import json
import platform
import sys

import numpy as np
import pandas as pd
import riskfolio
from pypfopt import HRPOpt
from scipy.optimize import linprog, nnls

from fixtures import FIXTURES, write_json

DATASETS = FIXTURES / "datasets.json"
OUTPUT = FIXTURES / "expected.json"

# Covariances better conditioned than this use the closed forms.
WELL_CONDITIONED = 1e12


def load_returns(dataset: dict) -> np.ndarray:
    data = pd.DataFrame(dataset["data"], dtype=float)
    if dataset["kind"] == "prices":
        data = data.pct_change().dropna()
    return data.to_numpy()


def adjust(cov: np.ndarray, shrinkage: float | None, regularization: float | None) -> np.ndarray:
    n = cov.shape[0]
    if shrinkage is not None:
        cov = shrinkage * (np.trace(cov) / n) * np.eye(n) + (1.0 - shrinkage) * cov
    if regularization is not None:
        cov = cov + regularization * np.eye(n)
    return cov


def well_conditioned(cov: np.ndarray) -> bool:
    return np.linalg.cond(cov) < WELL_CONDITIONED


def kkt_min_norm(k: np.ndarray, c: np.ndarray, e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-norm solution of [K E; Eᵀ 0][w; λ] = [c; b]."""
    n, m = e.shape
    kkt = np.block([[k, e], [e.T, np.zeros((m, m))]])
    solution, *_ = np.linalg.lstsq(kkt, np.concatenate([c, b]), rcond=None)
    return solution[:n]


def free_directions(cov: np.ndarray) -> np.ndarray:
    """Orthonormal basis of {d : Σd = 0, 1ᵀd = 0}: changes to a fully invested
    portfolio that add no variance."""
    n = cov.shape[0]
    stacked = np.vstack([cov / np.abs(cov).max(), np.ones((1, n))])
    _, singular, vt = np.linalg.svd(stacked)
    rank = int((singular > singular.max() * max(stacked.shape) * np.finfo(float).eps).sum())
    return vt[rank:].T


def min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    ones = np.ones(n)
    if well_conditioned(cov):
        inv_ones = np.linalg.solve(cov, ones)
        return inv_ones / inv_ones.sum()
    return kkt_min_norm(cov, np.zeros(n), ones[:, None], np.array([1.0]))


def long_only(k: np.ndarray, c: np.ndarray, e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimizes ½wᵀKw − cᵀw subject to Eᵀw = b and w ≥ 0 by support enumeration."""
    n, m = e.shape
    best, best_value = None, np.inf
    for mask in range(1, 2**n):
        support = [i for i in range(n) if mask >> i & 1]
        kkt = np.block([[k[np.ix_(support, support)], e[support]], [e[support].T, np.zeros((m, m))]])
        rhs = np.concatenate([c[support], b])
        solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
        # No feasible point on this support, such as a single asset away from the target.
        if np.abs(kkt @ solution - rhs).max() > 1e-9 * (np.abs(kkt).max() * np.abs(solution).max() + np.abs(rhs).max()):
            continue
        candidate = np.zeros(n)
        candidate[support] = solution[: len(support)]
        value = 0.5 * candidate @ k @ candidate - c @ candidate
        if candidate.min() >= 0.0 and value < best_value:
            best, best_value = candidate, value
    return best


def long_only_min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    return long_only(cov, np.zeros(n), np.ones((n, 1)), np.array([1.0]))


def long_only_mean_variance(cov: np.ndarray, mu: np.ndarray, params: dict) -> np.ndarray | str:
    n = cov.shape[0]
    if "risk_aversion" in params:
        return long_only(params["risk_aversion"] * cov, mu, np.ones((n, 1)), np.array([1.0]))
    target = params["target_return"]
    if not mu.min() <= target <= mu.max():
        return "TargetReturnOutOfRange"
    return long_only(cov, np.zeros(n), np.column_stack([np.ones(n), mu]), np.array([1.0, target]))


def risk_aversion(cov: np.ndarray, mu: np.ndarray, gamma: float) -> np.ndarray | str:
    n = cov.shape[0]
    ones = np.ones(n)
    if well_conditioned(cov):
        inv_ones = np.linalg.solve(cov, ones)
        inv_mu = np.linalg.solve(cov, mu)
        a, b = ones @ inv_ones, ones @ inv_mu
        eta = (b - gamma) / a
        return (inv_mu - eta * inv_ones) / gamma
    # Unbounded when some variance-free change to the portfolio raises expected return.
    if np.linalg.norm(free_directions(cov).T @ mu) > 1e-9 * np.linalg.norm(mu):
        return "Unbounded"
    return kkt_min_norm(gamma * cov, mu, ones[:, None], np.array([1.0]))


def target_return(cov: np.ndarray, mu: np.ndarray, target: float) -> np.ndarray | str:
    n = cov.shape[0]
    ones = np.ones(n)
    # With every expected return equal (relative to their size), only that return can
    # be targeted.
    scale = np.abs(mu).max()
    if np.ptp(mu) <= 1e-12 * scale and abs(target - mu.mean()) > 1e-12 * max(scale, abs(target)):
        return "DegenerateExpectedReturns"
    if well_conditioned(cov):
        inv_ones = np.linalg.solve(cov, ones)
        inv_mu = np.linalg.solve(cov, mu)
        a, b, c = ones @ inv_ones, ones @ inv_mu, mu @ inv_mu
        d = a * c - b * b
        return ((c - b * target) * inv_ones + (a * target - b) * inv_mu) / d
    return kkt_min_norm(cov, np.zeros(n), np.column_stack([ones, mu]), np.array([1.0, target]))


def max_sharpe(cov: np.ndarray, mu: np.ndarray, risk_free_rate: float) -> np.ndarray | str:
    """The tangency portfolio, or the error cutup should return.

    Expected errors are decided from first principles: the minimum variance portfolio (from
    min_variance above) and its variance and return, and the null space of Σ from NumPy's
    SVD. Nothing here runs cutup's algorithm.
    """
    n = cov.shape[0]
    excess = mu - risk_free_rate
    w_mvp = min_variance(cov)
    r_mvp = float(w_mvp @ mu)
    scale = np.trace(cov) / n

    # A fully invested portfolio with zero variance is the minimum variance portfolio.
    if w_mvp @ cov @ w_mvp <= 1e-12 * scale:
        return "Unbounded" if r_mvp > risk_free_rate else "RiskFreeRateTooHigh"
    # A variance-free, budget-neutral change that adds expected return.
    if np.linalg.norm(free_directions(cov).T @ excess) > 1e-9 * np.linalg.norm(excess):
        return "Unbounded"
    if risk_free_rate >= r_mvp - 1e-10 * (abs(r_mvp) + np.abs(mu).max()):
        return "RiskFreeRateTooHigh"
    z = np.linalg.solve(cov, excess) if well_conditioned(cov) else np.linalg.pinv(cov) @ excess
    return z / z.sum()


def long_only_max_sharpe(cov: np.ndarray, mu: np.ndarray, risk_free_rate: float) -> np.ndarray | str:
    n = cov.shape[0]
    excess = mu - risk_free_rate
    if excess.max() <= 1e-10 * max(abs(risk_free_rate), np.abs(mu).max()):
        return "NoPositiveExcessReturn"
    # The highest excess return of a long-only portfolio w = Nz in the null space N of Σ.
    values, vectors = np.linalg.eigh(cov)
    null = vectors[:, values <= n * np.finfo(float).eps * values.max()]
    if null.shape[1] > 0:
        zero_variance = linprog(
            -(excess @ null),
            A_ub=-null,
            b_ub=np.zeros(n),
            A_eq=(np.ones(n) @ null)[None, :],
            b_eq=[1.0],
            bounds=(None, None),
        )
        if zero_variance.status not in (0, 2):
            raise ValueError(f"linprog failed: {zero_variance.message}")
        if zero_variance.status == 0 and -zero_variance.fun > 1e-9 * np.abs(excess).max():
            return "Unbounded"
    y = long_only(cov, np.zeros(n), excess[:, None], np.array([1.0]))
    return y / y.sum()


def min_variance_return(cov: np.ndarray, mu: np.ndarray) -> float:
    return float(min_variance(cov) @ mu)


def hrp(returns: np.ndarray, linkage: str) -> np.ndarray:
    weights = HRPOpt(pd.DataFrame(returns)).optimize(linkage_method=linkage)
    return np.array([weights[i] for i in range(returns.shape[1])])


def risk_parity(cov: np.ndarray, budgets: np.ndarray) -> np.ndarray | str:
    """The risk budgeting portfolio, or the error cutup should return.

    Each coordinate step solves Σᵢᵢ yᵢ² + (Σy − Σᵢᵢ yᵢ)ᵢ yᵢ − bᵢ = 0 for yᵢ > 0, which
    minimizes ½ yᵀΣy − bᵀ ln y in that coordinate. When a long-only portfolio has zero
    variance there is no minimum.
    """
    n = cov.shape[0]
    cov = cov / (np.trace(cov) / n)
    budgets = budgets / budgets.sum()
    # min dᵀΣd over long-only d summing to 1, as a least-squares problem in Σ's square root.
    values, vectors = np.linalg.eigh(cov)
    root = np.sqrt(np.clip(values, 0.0, None))[:, None] * vectors.T
    weight = 1e4
    d, _ = nnls(np.vstack([root, weight * np.ones(n)]), np.r_[np.zeros(n), weight])
    if d @ cov @ d <= 1e-12:
        return "SingularCovariance"

    y = 1.0 / np.sqrt(np.diag(cov))
    y /= np.sqrt(y @ cov @ y)
    for _ in range(100_000):
        previous = y.copy()
        for i in range(n):
            c = cov[i] @ y - cov[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4.0 * cov[i, i] * budgets[i])) / (2.0 * cov[i, i])
        if np.max(np.abs(y - previous) / y) < 1e-14:
            return y / y.sum()
    raise ValueError("coordinate descent didn't converge")


def herc(returns: np.ndarray, linkage: str, clusters: int | None) -> np.ndarray:
    data = pd.DataFrame(returns, columns=[str(i) for i in range(returns.shape[1])])
    weights = riskfolio.HCPortfolio(returns=data).optimization(
        model="HERC", codependence="pearson", rm="MV", linkage=linkage, k=clusters
    )
    return weights["weights"].to_numpy()


def run(case: dict, returns: np.ndarray, adjusted: np.ndarray) -> np.ndarray | str:
    params = case.get("params", {})
    mu = np.array(params["expected_returns"]) if "expected_returns" in params else returns.mean(axis=0)
    match case["method"]:
        case "equal_weight":
            return np.full(returns.shape[1], 1.0 / returns.shape[1])
        case "inverse_volatility":
            raw = 1.0 / np.sqrt(np.var(returns, axis=0, ddof=1))
            return raw / raw.sum()
        case "inverse_variance":
            raw = 1.0 / np.var(returns, axis=0, ddof=1)
            return raw / raw.sum()
        case "min_variance":
            return min_variance(adjusted)
        case "long_only_min_variance":
            return long_only_min_variance(adjusted)
        case "long_only_mean_variance":
            return long_only_mean_variance(adjusted, mu, params)
        case "mean_variance" if "risk_aversion" in params:
            return risk_aversion(adjusted, mu, params["risk_aversion"])
        case "mean_variance":
            return target_return(adjusted, mu, params["target_return"])
        case "max_sharpe":
            return max_sharpe(adjusted, mu, params.get("risk_free_rate", 0.0))
        case "long_only_max_sharpe":
            return long_only_max_sharpe(adjusted, mu, params.get("risk_free_rate", 0.0))
        case "risk_parity":
            budgets = np.array(params.get("risk_budgets", np.ones(returns.shape[1])), dtype=float)
            return risk_parity(adjusted, budgets)
        case "hrp":
            return hrp(returns, params.get("linkage", "single"))
        case "herc":
            return herc(returns, params.get("linkage", "single"), params.get("clusters"))
    raise ValueError(f"unknown method {case['method']}")


def cases(datasets: dict) -> list[dict]:
    """The (dataset, method, params) combinations to check, with tolerances."""
    exact, solver, clustering = 0.0, 1e-9, 1e-12
    out = []
    for name in datasets:
        out.append({"dataset": name, "method": "risk_parity", "tolerance": solver})
        out.append({"dataset": name, "method": "equal_weight", "tolerance": exact})
        out.append({"dataset": name, "method": "min_variance", "tolerance": solver})
        out.append({"dataset": name, "method": "hrp", "tolerance": clustering})
        out.append({"dataset": name, "method": "inverse_volatility", "tolerance": clustering})
        out.append({"dataset": name, "method": "inverse_variance", "tolerance": clustering})
        for linkage in ["complete", "average", "ward"]:
            out.append({"dataset": name, "method": "hrp", "params": {"linkage": linkage}, "tolerance": clustering})

    for name in ["two_blocks5", "prices10x60", "near_singular4", "rank_deficient6"]:
        out.append({"dataset": name, "method": "min_variance", "params": {"shrinkage": 0.3}, "tolerance": solver})

    # The long-only optimum is unique only when Σ isn't singular, so the singular datasets
    # are used with shrinkage.
    for name in ["orthogonal3", "two_asset", "two_blocks5", "prices10x60", "near_singular4"]:
        out.append({"dataset": name, "method": "long_only_min_variance", "tolerance": solver})
    for name in ["two_blocks5", "prices10x60", "near_singular4", "rank_deficient6", "collinear2"]:
        out.append({"dataset": name, "method": "long_only_min_variance", "params": {"shrinkage": 0.3}, "tolerance": solver})
    for name in ["prices10x60", "near_singular4"]:
        out.append({"dataset": name, "method": "long_only_min_variance", "params": {"regularization": 1e-5}, "tolerance": solver})
    for name in ["prices10x60", "near_singular4"]:
        out.append({"dataset": name, "method": "min_variance", "params": {"regularization": 1e-5}, "tolerance": solver})

    # Long-only mean-variance, like long-only min variance, only on covariances that aren't
    # singular. Low risk aversions and targets near the highest return make the constraint bind.
    for name, gamma in [("orthogonal3", 50.0), ("two_asset", 50.0), ("two_blocks5", 100.0),
                        ("prices10x60", 100.0), ("prices10x60", 5.0), ("near_singular4", 100.0)]:
        out.append({"dataset": name, "method": "long_only_mean_variance", "params": {"risk_aversion": gamma}, "tolerance": solver})
    for name in ["rank_deficient6", "collinear2"]:
        out.append({
            "dataset": name,
            "method": "long_only_mean_variance",
            "params": {"risk_aversion": 100.0, "shrinkage": 0.3},
            "tolerance": solver,
        })
    out.append({
        "dataset": "prices10x60",
        "method": "long_only_mean_variance",
        "params": {
            "risk_aversion": 100.0,
            "expected_returns": [0.0010, 0.0004, 0.0007, 0.0002, 0.0012, 0.0005, 0.0003, 0.0008, 0.0006, 0.0009],
        },
        "tolerance": solver,
    })
    for name, fraction in [("two_asset", 0.5), ("two_blocks5", 0.75), ("prices10x60", 0.5),
                           ("prices10x60", 0.9), ("near_singular4", 0.75)]:
        mu = load_returns(datasets[name]).mean(axis=0)
        target = round(float(mu.min() + fraction * (mu.max() - mu.min())), 6)
        out.append({"dataset": name, "method": "long_only_mean_variance", "params": {"target_return": target}, "tolerance": solver})
    out.append({
        "dataset": "rank_deficient6",
        "method": "long_only_mean_variance",
        "params": {"target_return": 0.005, "shrinkage": 0.3},
        "tolerance": solver,
    })
    # Every expected return is zero, so only zero can be targeted.
    out.append({"dataset": "orthogonal3", "method": "long_only_mean_variance", "params": {"target_return": 0.0}, "tolerance": solver})
    out.append({"dataset": "orthogonal3", "method": "long_only_mean_variance", "params": {"target_return": 0.01}, "tolerance": solver})
    out.append({"dataset": "prices10x60", "method": "long_only_mean_variance", "params": {"target_return": 0.01}, "tolerance": solver})
    out.append({
        "dataset": "near_singular4",
        "method": "min_variance",
        "params": {"shrinkage": 0.2, "regularization": 1e-6},
        "tolerance": solver,
    })

    for name, gamma in [("orthogonal3", 50.0), ("two_asset", 50.0), ("two_blocks5", 100.0),
                        ("prices10x60", 100.0), ("near_singular4", 100.0), ("rank_deficient6", 100.0)]:
        out.append({"dataset": name, "method": "mean_variance", "params": {"risk_aversion": gamma}, "tolerance": solver})
    out.append({
        "dataset": "rank_deficient6",
        "method": "mean_variance",
        "params": {"risk_aversion": 100.0, "shrinkage": 0.3},
        "tolerance": solver,
    })
    out.append({
        "dataset": "prices10x60",
        "method": "mean_variance",
        "params": {
            "risk_aversion": 100.0,
            "expected_returns": [0.0010, 0.0004, 0.0007, 0.0002, 0.0012, 0.0005, 0.0003, 0.0008, 0.0006, 0.0009],
        },
        "tolerance": solver,
    })

    for name in ["two_asset", "two_blocks5", "prices10x60", "near_singular4", "rank_deficient6"]:
        returns = load_returns(datasets[name])
        target = round(1.25 * float(returns.mean()), 6)
        out.append({"dataset": name, "method": "mean_variance", "params": {"target_return": target}, "tolerance": solver})
    out.append({"dataset": "orthogonal3", "method": "mean_variance", "params": {"target_return": 0.01}, "tolerance": solver})

    # Max Sharpe at a risk-free rate just below the minimum variance return, where the
    # maximum exists, plus the error cases.
    for name in ["two_asset", "two_blocks5", "prices10x60", "near_singular4"]:
        returns = load_returns(datasets[name])
        threshold = min_variance_return(np.cov(returns, rowvar=False), returns.mean(axis=0))
        rate = round(threshold - abs(threshold) * 0.5 - 1e-4, 6)
        out.append({"dataset": name, "method": "max_sharpe", "params": {"risk_free_rate": rate}, "tolerance": solver})
    out.append({"dataset": "prices10x60", "method": "max_sharpe", "params": {"risk_free_rate": 0.0, "shrinkage": 0.3}, "tolerance": solver})
    out.append({
        "dataset": "prices10x60",
        "method": "max_sharpe",
        "params": {
            "risk_free_rate": 0.0001,
            "expected_returns": [0.0010, 0.0004, 0.0007, 0.0002, 0.0012, 0.0005, 0.0003, 0.0008, 0.0006, 0.0009],
        },
        "tolerance": solver,
    })
    out.append({"dataset": "prices10x60", "method": "max_sharpe", "params": {"risk_free_rate": 0.01}, "tolerance": solver})
    out.append({"dataset": "rank_deficient6", "method": "max_sharpe", "tolerance": solver})
    out.append({"dataset": "rank_deficient6", "method": "max_sharpe", "params": {"risk_free_rate": -0.01, "shrinkage": 0.3}, "tolerance": solver})
    # The collinear pair's zero-variance portfolio returns −0.001: below it, the Sharpe ratio
    # is unbounded; above it, the rate is too high.
    out.append({"dataset": "collinear2", "method": "max_sharpe", "params": {"risk_free_rate": 0.0}, "tolerance": solver})
    out.append({"dataset": "collinear2", "method": "max_sharpe", "params": {"risk_free_rate": -0.01}, "tolerance": solver})
    out.append({"dataset": "collinear2", "method": "max_sharpe", "params": {"risk_free_rate": 0.0, "shrinkage": 0.3}, "tolerance": solver})

    # Long-only max Sharpe at rates below every expected return and at the middle of their
    # range. On two_asset and prices10x60 the middle, like 0.004 on prices10x60, is between
    # the minimum variance return and the highest expected return, where only the long-only
    # maximum exists.
    for name in ["two_asset", "two_blocks5", "prices10x60", "near_singular4"]:
        mu = load_returns(datasets[name]).mean(axis=0)
        for fraction in [-0.5, 0.5]:
            rate = round(float(mu.min() + fraction * (mu.max() - mu.min())), 6)
            out.append({"dataset": name, "method": "long_only_max_sharpe", "params": {"risk_free_rate": rate}, "tolerance": solver})
    out.append({"dataset": "orthogonal3", "method": "long_only_max_sharpe", "params": {"risk_free_rate": -0.01}, "tolerance": solver})
    out.append({"dataset": "prices10x60", "method": "long_only_max_sharpe", "params": {"risk_free_rate": 0.004}, "tolerance": solver})
    out.append({
        "dataset": "prices10x60",
        "method": "long_only_max_sharpe",
        "params": {
            "risk_free_rate": 0.0001,
            "expected_returns": [0.0010, 0.0004, 0.0007, 0.0002, 0.0012, 0.0005, 0.0003, 0.0008, 0.0006, 0.0009],
        },
        "tolerance": solver,
    })
    for name in ["rank_deficient6", "collinear2"]:
        out.append({"dataset": name, "method": "long_only_max_sharpe", "params": {"shrinkage": 0.3}, "tolerance": solver})
        out.append({"dataset": name, "method": "long_only_max_sharpe", "params": {"risk_free_rate": -0.01}, "tolerance": solver})
    out.append({"dataset": "prices10x60", "method": "long_only_max_sharpe", "params": {"risk_free_rate": 0.01}, "tolerance": solver})
    out.append({"dataset": "orthogonal3", "method": "long_only_max_sharpe", "tolerance": solver})

    # rank_deficient6 has a long-only portfolio with zero variance, so it only has a risk
    # parity portfolio once shrinkage removes it.
    out.append({"dataset": "rank_deficient6", "method": "risk_parity", "params": {"shrinkage": 0.3}, "tolerance": solver})
    out.append({"dataset": "near_singular4", "method": "risk_parity", "params": {"regularization": 1e-5}, "tolerance": solver})
    out.append({
        "dataset": "prices10x60",
        "method": "risk_parity",
        "params": {"risk_budgets": [4.0, 1.0, 1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 1.0, 0.25]},
        "tolerance": solver,
    })
    out.append({"dataset": "two_asset", "method": "risk_parity", "params": {"risk_budgets": [3.0, 1.0]}, "tolerance": solver})

    # riskfolio-lib can't choose a number of clusters for two assets.
    for name in [name for name, dataset in datasets.items() if len(dataset["data"][0]) > 2]:
        for linkage in ["single", "complete", "average", "ward"]:
            out.append({"dataset": name, "method": "herc", "params": {"linkage": linkage}, "tolerance": clustering})
    for clusters in [1, 2, 4, 10]:
        out.append({"dataset": "prices10x60", "method": "herc", "params": {"clusters": clusters}, "tolerance": clustering})
    out.append({"dataset": "prices10x60", "method": "herc", "params": {"linkage": "ward", "clusters": 5}, "tolerance": clustering})
    out.append({"dataset": "two_blocks5", "method": "herc", "params": {"linkage": "average", "clusters": 3}, "tolerance": clustering})
    return out


def drift(committed: dict, fresh: dict) -> list[str]:
    """Differences between the committed fixtures and a fresh run that exceed noise."""
    problems = []
    if len(committed["cases"]) != len(fresh["cases"]):
        return [f"{len(committed['cases'])} committed cases, {len(fresh['cases'])} regenerated"]
    for old, new in zip(committed["cases"], fresh["cases"]):
        label = f"{new['dataset']} / {new['method']} {new.get('params', {})}"
        keys = ("dataset", "method", "params", "error")
        if any(old.get(key) != new.get(key) for key in keys) or ("weights" in old) != ("weights" in new):
            problems.append(f"{label}: case changed")
            continue
        # Tolerances come from the condition number, which carries platform noise, so
        # only a change of more than 2x counts as a different tolerance rule.
        old_tol, new_tol = old["tolerance"], new["tolerance"]
        if (old_tol == 0) != (new_tol == 0) or (old_tol and not 0.5 <= new_tol / old_tol <= 2.0):
            problems.append(f"{label}: tolerance changed from {old_tol} to {new_tol}")
        if "weights" not in new:
            continue
        old_w, new_w = np.array(old["weights"]), np.array(new["weights"])
        if old_w.shape != new_w.shape:
            problems.append(f"{label}: {old_w.size} committed weights, {new_w.size} regenerated")
            continue
        allowed = max(old["tolerance"], new["tolerance"]) / 10
        moved = np.max(np.abs(old_w - new_w) / np.maximum(np.abs(old_w), 1.0))
        # Written as "not <=" so that a NaN counts as drift.
        if not moved <= allowed:
            problems.append(f"{label}: weights moved by {moved:.2e} (allowed {allowed:.0e})")
    return problems


def main() -> None:
    check = "--check" in sys.argv[1:]
    datasets = json.loads(DATASETS.read_text())
    returns_by_dataset = {name: load_returns(dataset) for name, dataset in datasets.items()}
    results = []
    for case in cases(datasets):
        returns = returns_by_dataset[case["dataset"]]
        params = case.get("params", {})
        adjusted = adjust(np.cov(returns, rowvar=False), params.get("shrinkage"), params.get("regularization"))
        outcome = run(case, returns, adjusted)
        if isinstance(outcome, str):
            case["error"] = outcome
        elif not np.all(np.isfinite(outcome)):
            raise ValueError(f"non-finite reference weights for {case}: {outcome}")
        else:
            case["weights"] = [float(w) for w in outcome]
        if case["method"] in ("min_variance", "long_only_min_variance", "long_only_mean_variance", "long_only_max_sharpe", "mean_variance", "max_sharpe"):
            # A numerically singular covariance has its null space truncated, which
            # every solver does alike. Short of that, any solver loses about cond(Σ) · ε
            # of accuracy, so ill-conditioned cases get a proportionally looser tolerance.
            condition = float(np.linalg.cond(adjusted))
            case["condition_number"] = float(f"{condition:.3e}")
            singular = condition > 1.0 / (adjusted.shape[0] * np.finfo(float).eps)
            if not singular:
                case["tolerance"] = max(case["tolerance"], float(f"{condition * 1e-14:.1e}"))
        results.append(case)

    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "pyportfolioopt", "riskfolio-lib"]:
        versions[package] = importlib.metadata.version(package)
    fresh = {"generated_with": versions, "cases": results}

    if check:
        problems = drift(json.loads(OUTPUT.read_text()), fresh)
        for problem in problems:
            print(problem)
        print(f"{len(problems)} of {len(results)} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)

    write_json(OUTPUT, fresh)
    errors = sum("error" in case for case in results)
    print(f"wrote {len(results)} cases ({errors} expected errors) to {OUTPUT}")


if __name__ == "__main__":
    main()
