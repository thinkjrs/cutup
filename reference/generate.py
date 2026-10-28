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

  Every covariance-based reference works in correlation units u = diag(σ)w, as cutup
  does: Σ = diag(σ) C diag(σ), linear solves and pseudo-inverses go through the
  correlation matrix C, minimum norms are of u, and null spaces and condition numbers are
  C's. Then nothing depends on each asset's units, and assets whose volatilities are many
  orders of magnitude apart stay accurate.
* Long-only minimum variance and mean-variance: support enumeration. On every subset of
  assets, the KKT system with the others held at zero gives that subset's optimum, if it
  is feasible. Among the candidates with no negative weights, the best objective is the
  long-only optimum, since each candidate is feasible and the optimum is one of them. This
  checks neither the multipliers' signs nor uses an iterative solver, unlike cutup's
  active-set method. Only covariances that aren't singular are used, where the optimum is
  unique. A target return outside the assets' expected returns is an error.

  With weight bounds, each asset is held at its lower bound, at its upper bound, or left
  free. The range of reachable target returns comes from scipy.optimize.linprog, not the
  greedy fill cutup uses, and bounds that can't sum to 1 are an error. Each bounded result
  is cross-checked against PyPortfolioOpt's EfficientFrontier with the same weight_bounds,
  which uses cvxpy, to 1e-5.
* Long-only max Sharpe: the same support enumeration, on the change of variables
  y = w / (μ − r_f)ᵀw that turns it into minimizing yᵀΣy subject to (μ − r_f)ᵀy = 1 and
  y ≥ 0. An unbounded Sharpe ratio is decided by scipy.optimize.linprog finding a
  long-only portfolio in the correlation matrix's null space (eigenvalues within n·ε of
  the largest, from numpy.linalg.eigh) with positive excess return.
* Max diversification: the same support enumeration and linear program as long-only max
  Sharpe, with the volatilities in place of the excess returns.
* Max Sharpe: the NumPy closed form Σ⁻¹(μ − r_f) / 1ᵀΣ⁻¹(μ − r_f); the minimum-norm
  least-squares solution when Σ is singular.
* Risk parity: cyclical coordinate descent (Griveau-Billion et al., 2013) run to
  convergence, a different algorithm from cutup's Newton iteration. Expected errors are
  decided by finding a long-only portfolio with zero variance with scipy.optimize.nnls.
* HRP: PyPortfolioOpt's HRPOpt, for each linkage method.
* HERC: riskfolio-lib's HCPortfolio with model="HERC" and rm="MV", for each linkage
  method and for fixed numbers of clusters.
* Turnover and transaction costs, for min variance and mean-variance from current weights
  w₀: support enumeration with five states per asset, at its lower bound, at its upper
  bound, at w₀, or free above or below w₀, where the cost cᵢ|wᵢ − w₀ᵢ| is the linear term
  ±cᵢ. A turnover cap adds, on each support, a candidate with the cap as an equality
  Σ sign(wᵢ − w₀ᵢ)(wᵢ − w₀ᵢ) = T. The best feasible candidate is the optimum, as above. The
  objective is written out from each method's definition, not from cutup's. Each result
  is cross-checked against PyPortfolioOpt, with an L1 cost, a squared-distance objective
  and an L1 constraint on cvxpy with Clarabel at tight tolerances, to 1e-6. A cap below the turnover needed to
  move w₀ into the bounds is an expected error.
* Black-Litterman: PyPortfolioOpt's market_implied_prior_returns and
  BlackLittermanModel.bl_returns, with its "default" (He-Litterman) and "idzorek" view
  uncertainties, or, when a case mixes kinds of view, a diagonal Ω built row by row with
  BlackLittermanModel.default_omega and idzorek_method. These cases hold expected returns,
  not weights.

With --check, nothing is written: the cases are recomputed and compared with the
committed file, failing if any weight moved by more than a tenth of its tolerance or
any case changed shape. CI runs this so the fixtures can't drift from the pinned
reference implementations. The comparison is numeric rather than byte-for-byte
because NumPy's linear algebra can differ in the last bits between platforms.
"""

import importlib.metadata
import itertools
import json
import platform
import sys

import cvxpy as cp
import numpy as np
import pandas as pd
import riskfolio
from pypfopt import EfficientFrontier, HRPOpt
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
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


def correlation(cov: np.ndarray) -> np.ndarray:
    volatilities = np.sqrt(np.diag(cov))
    return cov / volatilities[:, None] / volatilities[None, :]


def unit_scales(k: np.ndarray) -> np.ndarray:
    """sqrt(diag K), or 1 where that isn't positive: u = Dx gives K a unit diagonal."""
    scales = np.sqrt(np.clip(np.diag(k), 0.0, None))
    return np.where(scales > 0.0, scales, 1.0)


def well_conditioned(cov: np.ndarray) -> bool:
    return np.linalg.cond(correlation(cov)) < WELL_CONDITIONED


def solve(cov: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Σ⁻¹x = D⁻¹C⁻¹D⁻¹x."""
    d = np.sqrt(np.diag(cov))
    return np.linalg.solve(correlation(cov), x / d) / d


def pinv(cov: np.ndarray, x: np.ndarray) -> np.ndarray:
    """D⁻¹C⁺D⁻¹x, the minimum-norm solution in correlation units."""
    d = np.sqrt(np.diag(cov))
    return np.linalg.pinv(correlation(cov)) @ (x / d) / d


def kkt_min_norm(k: np.ndarray, c: np.ndarray, e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-norm solution of [K E; Eᵀ 0][w; λ] = [c; b], in units u = Dw with
    D = diag(sqrt(Kᵢᵢ)). Each column of E is first divided by its largest entry, with b."""
    n, m = e.shape
    d = unit_scales(k)
    k, c, e, b = in_units(k, c, e, b, d)
    kkt = np.block([[k, e], [e.T, np.zeros((m, m))]])
    solution, *_ = np.linalg.lstsq(kkt, np.concatenate([c, b]), rcond=None)
    return solution[:n] / d


def in_units(k: np.ndarray, c: np.ndarray, e: np.ndarray, b: np.ndarray, d: np.ndarray):
    """The problem in u = Dx, with each column of E and its entry of b divided by the
    column's largest entry."""
    columns = np.abs(e).max(axis=0)
    columns = np.where(columns > 0.0, columns, 1.0)
    return k / d[:, None] / d[None, :], c / d, e / columns / d[:, None], b / columns


def free_fraction(cov: np.ndarray, v: np.ndarray) -> float:
    """How much of v, relative to its size, a change to a fully invested portfolio that
    adds no variance can pick up: in correlation units, the norm of D⁻¹v's projection onto
    {f : Cf = 0, (D⁻¹1)ᵀf = 0} over the norm of D⁻¹v."""
    d = np.sqrt(np.diag(cov))
    free = free_directions(correlation(cov), 1.0 / d)
    return float(np.linalg.norm(free.T @ (v / d)) / np.linalg.norm(v / d))


def free_directions(corr: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """Orthonormal basis of {f : Cf = 0, budgetᵀf = 0}."""
    stacked = np.vstack([corr, budget / np.abs(budget).max()])
    _, singular, vt = np.linalg.svd(stacked)
    rank = int((singular > singular.max() * max(stacked.shape) * np.finfo(float).eps).sum())
    return vt[rank:].T


def min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    ones = np.ones(n)
    if well_conditioned(cov):
        inv_ones = solve(cov, ones)
        return inv_ones / inv_ones.sum()
    return kkt_min_norm(cov, np.zeros(n), ones[:, None], np.array([1.0]))


def long_only(
    k: np.ndarray,
    c: np.ndarray,
    e: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray | None:
    """Minimizes ½wᵀKw − cᵀw subject to Eᵀw = b and l ≤ w ≤ u, by default w ≥ 0, by support
    enumeration: every variable is held at its lower bound, at its upper bound, or left free.
    Works in units u = Dw with D = diag(sqrt(Kᵢᵢ)), which change neither the objective nor
    which points are within the bounds."""
    n, m = e.shape
    lower = np.zeros(n) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float)
    d = unit_scales(k)
    k, c, e, b = in_units(k, c, e, b, d)
    lower, upper = lower * d, upper * d
    # 0: at the lower bound, 1: at the upper bound, 2: free.
    choices = [
        (0,) if upper[i] <= lower[i] else (0, 2) if np.isinf(upper[i]) else (0, 1, 2)
        for i in range(n)
    ]
    best, best_value = None, np.inf
    for states in itertools.product(*choices):
        support = [i for i in range(n) if states[i] == 2]
        held = [i for i in range(n) if states[i] != 2]
        candidate = np.zeros(n)
        for i in held:
            candidate[i] = upper[i] if states[i] == 1 else lower[i]
        # The free variables' problem, with the held ones moved to the right-hand side.
        c_s = c[support] - k[np.ix_(support, held)] @ candidate[held]
        b_s = b - e[held].T @ candidate[held]
        if not support:
            # No feasible point with every variable at a bound here, such as a single
            # asset away from the target.
            if np.abs(b_s).max() > 1e-12 * (np.abs(e).max() * np.abs(candidate).max() + np.abs(b).max()):
                continue
        else:
            k_s, e_s = k[np.ix_(support, support)], e[support]
            if np.linalg.cond(k_s) < WELL_CONDITIONED:
                # Kx = c − Eλ, so the multipliers solve (EᵀK⁻¹E)λ = EᵀK⁻¹c − b. Unlike a
                # least-squares solve of the whole KKT matrix, this stays accurate when E's
                # entries are many orders of magnitude apart.
                k_e, k_c = np.linalg.solve(k_s, e_s), np.linalg.solve(k_s, c_s)
                lam, *_ = np.linalg.lstsq(e_s.T @ k_e, e_s.T @ k_c - b_s, rcond=None)
                x = k_c - k_e @ lam
                residual, size = e_s.T @ x - b_s, np.abs(e_s).max() * np.abs(x).max() + np.abs(b).max()
            else:
                kkt = np.block([[k_s, e_s], [e_s.T, np.zeros((m, m))]])
                rhs = np.concatenate([c_s, b_s])
                solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
                x = solution[: len(support)]
                residual, size = kkt @ solution - rhs, np.abs(kkt).max() * np.abs(solution).max() + np.abs(rhs).max()
            # No feasible point on this support, such as a single asset away from the target.
            if np.abs(residual).max() > 1e-9 * size:
                continue
            if np.any(x < lower[support]) or np.any(x > upper[support]):
                continue
            candidate[support] = x
        value = 0.5 * candidate @ k @ candidate - c @ candidate
        if value < best_value:
            best, best_value = candidate, value
    # None when no support is feasible, such as an unreachable target return.
    return None if best is None else best / d


def weight_bounds(params: dict, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The case's weight bounds, from `bounds` for every asset or per-asset `lower_bounds` and
    `upper_bounds`, or None."""
    if "bounds" in params:
        lower, upper = params["bounds"]
        return np.full(n, float(lower)), np.full(n, float(upper))
    if "lower_bounds" in params:
        return np.array(params["lower_bounds"], dtype=float), np.array(params["upper_bounds"], dtype=float)
    return None


def infeasible(bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    lower, upper = bounds
    return lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12


def return_range(mu: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """The lowest and highest expected return of a portfolio summing to 1 within the bounds,
    from two linear programs."""
    lower, upper = bounds
    box = list(zip(lower, upper))
    budget = {"A_eq": np.ones((1, len(mu))), "b_eq": [1.0], "bounds": box}
    lowest, highest = linprog(mu, **budget), linprog(-mu, **budget)
    if lowest.status != 0 or highest.status != 0:
        raise ValueError(f"linprog failed: {lowest.message} / {highest.message}")
    return float(lowest.fun), float(-highest.fun)


def long_only_min_variance(cov: np.ndarray, params: dict) -> np.ndarray | str:
    n = cov.shape[0]
    bounds = weight_bounds(params, n)
    if bounds is None:
        return long_only(cov, np.zeros(n), np.ones((n, 1)), np.array([1.0]))
    if infeasible(bounds):
        return "InfeasibleBounds"
    weights = long_only(cov, np.zeros(n), np.ones((n, 1)), np.array([1.0]), *bounds)
    cross_check(cov, None, bounds, weights, lambda frontier: frontier.min_volatility())
    return weights


def long_only_mean_variance(cov: np.ndarray, mu: np.ndarray, params: dict) -> np.ndarray | str:
    n = cov.shape[0]
    bounds = weight_bounds(params, n)
    if bounds is not None and infeasible(bounds):
        return "InfeasibleBounds"
    bounds_or_none = bounds or (None, None)
    if "risk_aversion" in params:
        gamma = params["risk_aversion"]
        weights = long_only(gamma * cov, mu, np.ones((n, 1)), np.array([1.0]), *bounds_or_none)
        if bounds is not None:
            cross_check(cov, mu, bounds, weights, lambda frontier: frontier.max_quadratic_utility(gamma))
        return weights
    target = params["target_return"]
    lowest, highest = (mu.min(), mu.max()) if bounds is None else return_range(mu, bounds)
    tolerance = 1e-12 * max(abs(lowest), abs(highest))
    if not lowest - tolerance <= target <= highest + tolerance:
        return "TargetReturnOutOfRange"
    target = min(max(target, lowest), highest)
    weights = long_only(cov, np.zeros(n), np.column_stack([np.ones(n), mu]), np.array([1.0, target]), *bounds_or_none)
    # PyPortfolioOpt only accepts targets strictly inside the range.
    if bounds is not None and lowest < target < highest:
        cross_check(cov, mu, bounds, weights, lambda frontier: frontier.efficient_return(target))
    return weights


def cross_check(cov, mu, bounds, weights, optimize) -> None:
    """Checks bounded weights against PyPortfolioOpt's EfficientFrontier with the same
    weight bounds, which solves the problem with cvxpy. That solver is only accurate to
    about 1e-6, so this catches a wrong reference rather than setting the fixture. It's
    skipped when Σ itself is badly conditioned, as when volatilities are orders of
    magnitude apart: cvxpy's absolute tolerances can't tell such variances apart."""
    if np.linalg.cond(cov) > 1e8:
        return
    lower, upper = bounds
    frontier = EfficientFrontier(mu, cov, weight_bounds=list(zip(lower, np.minimum(upper, 1.0))))
    optimize(frontier)
    theirs = np.array(list(frontier.weights))
    # efficient_return asks for a return of at least the target, which only matches an
    # equality where the target is above the minimum variance portfolio's return.
    if mu is not None and abs(theirs @ mu - weights @ mu) > 1e-9 * np.abs(mu).max():
        return
    if np.abs(theirs - weights).max() > 1e-5:
        raise ValueError(f"PyPortfolioOpt disagrees: {theirs} vs {weights}")


def risk_aversion(cov: np.ndarray, mu: np.ndarray, gamma: float) -> np.ndarray | str:
    n = cov.shape[0]
    ones = np.ones(n)
    if well_conditioned(cov):
        inv_ones = solve(cov, ones)
        inv_mu = solve(cov, mu)
        a, b = ones @ inv_ones, ones @ inv_mu
        eta = (b - gamma) / a
        return (inv_mu - eta * inv_ones) / gamma
    # Unbounded when some variance-free change to the portfolio raises expected return.
    if free_fraction(cov, mu) > 1e-9:
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
        inv_ones = solve(cov, ones)
        inv_mu = solve(cov, mu)
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
    u_mvp = np.sqrt(np.diag(cov)) * w_mvp

    # A fully invested portfolio with zero variance is the minimum variance portfolio.
    if u_mvp @ correlation(cov) @ u_mvp <= 1e-12 * (u_mvp @ u_mvp):
        return "Unbounded" if r_mvp > risk_free_rate else "RiskFreeRateTooHigh"
    # A variance-free, budget-neutral change that adds expected return.
    if free_fraction(cov, excess) > 1e-9:
        return "Unbounded"
    if risk_free_rate >= r_mvp - 1e-10 * np.abs(w_mvp * mu).sum():
        return "RiskFreeRateTooHigh"
    z = solve(cov, excess) if well_conditioned(cov) else pinv(cov, excess)
    return z / z.sum()


def long_only_max_ratio(cov: np.ndarray, a: np.ndarray) -> np.ndarray | None:
    """Maximizes aᵀw / sqrt(wᵀΣw) over long-only w summing to 1, for a with a positive
    entry, or None when the ratio is unbounded."""
    n = cov.shape[0]
    # The highest aᵀw of a long-only portfolio w = D⁻¹Nz, for N the null space of the
    # correlation matrix: in correlation units u = Dw, the objective is (D⁻¹a)ᵀu and the
    # budget (D⁻¹1)ᵀu, each divided by its largest entry.
    d = np.sqrt(np.diag(cov))
    values, vectors = np.linalg.eigh(correlation(cov))
    null = vectors[:, values <= n * np.finfo(float).eps * values.max()]
    if null.shape[1] > 0:
        earned, budget = a / d, 1.0 / d
        zero_variance = linprog(
            -(earned / np.abs(earned).max()) @ null,
            A_ub=-null,
            b_ub=np.zeros(n),
            A_eq=((budget / budget.max()) @ null)[None, :],
            b_eq=[1.0],
            bounds=(None, None),
        )
        if zero_variance.status not in (0, 2):
            raise ValueError(f"linprog failed: {zero_variance.message}")
        if zero_variance.status == 0:
            w = null @ zero_variance.x / d
            if a @ (w / w.sum()) > 1e-9 * np.abs(a).max():
                return None
    y = long_only(cov, np.zeros(n), a[:, None], np.array([1.0]))
    return y / y.sum()


def long_only_max_sharpe(cov: np.ndarray, mu: np.ndarray, risk_free_rate: float) -> np.ndarray | str:
    excess = mu - risk_free_rate
    if np.all(excess <= 1e-10 * np.maximum(abs(risk_free_rate), np.abs(mu))):
        return "NoPositiveExcessReturn"
    weights = long_only_max_ratio(cov, excess)
    return "Unbounded" if weights is None else weights


def max_diversification(cov: np.ndarray) -> np.ndarray | str:
    weights = long_only_max_ratio(cov, np.sqrt(np.diag(cov)))
    return "SingularCovariance" if weights is None else weights


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

    It runs on the correlation matrix in units u = diag(σ)y, where the risk contributions
    are the same, and divides by the volatilities at the end.
    """
    n = cov.shape[0]
    volatilities = np.sqrt(np.diag(cov))
    cov = correlation(cov)
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
            w = y / volatilities
            return w / w.sum()
    raise ValueError("coordinate descent didn't converge")


def herc(returns: np.ndarray, linkage: str, clusters: int | None) -> np.ndarray:
    data = pd.DataFrame(returns, columns=[str(i) for i in range(returns.shape[1])])
    weights = riskfolio.HCPortfolio(returns=data).optimization(
        model="HERC", codependence="pearson", rm="MV", linkage=linkage, k=clusters
    )
    return weights["weights"].to_numpy()


def kinked(
    k: np.ndarray,
    c: np.ndarray,
    current: np.ndarray,
    costs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cap: float | None,
) -> np.ndarray | None:
    """Minimizes ½wᵀKw − cᵀw + Σᵢ costsᵢ|wᵢ − currentᵢ| subject to 1ᵀw = 1, l ≤ w ≤ u and,
    with a cap, Σᵢ|wᵢ − currentᵢ| ≤ cap, by support enumeration over five states per asset."""
    n = len(c)
    # 0: lower bound, 1: upper bound, 2: at current, 3: free above it, 4: free below it.
    def choices(i):
        out = []
        if np.isfinite(lower[i]):
            out.append(0)
        if np.isfinite(upper[i]) and upper[i] > lower[i]:
            out.append(1)
        if lower[i] <= current[i] <= upper[i]:
            out.append(2)
        if current[i] < upper[i] and upper[i] > lower[i]:
            out.append(3)
        if current[i] > lower[i] and upper[i] > lower[i]:
            out.append(4)
        return out

    def objective(w):
        return 0.5 * w @ k @ w - c @ w + costs @ np.abs(w - current)

    best, best_value = None, np.inf
    for states in itertools.product(*(choices(i) for i in range(n))):
        free = [i for i in range(n) if states[i] >= 3]
        held = [i for i in range(n) if states[i] < 3]
        w = np.zeros(n)
        for i in held:
            w[i] = (lower[i], upper[i], current[i])[states[i]]
        sign = np.array([1.0 if states[i] == 3 else -1.0 for i in free])
        for capped in ([False, True] if cap is not None else [False]):
            if not free:
                candidates = [w.copy()]
            else:
                columns = [np.ones(len(free))]
                rhs = [1.0 - w[held].sum()]
                if capped:
                    columns.append(sign)
                    rhs.append(cap - np.abs(w[held] - current[held]).sum() + sign @ current[free])
                e = np.column_stack(columns)
                m = e.shape[1]
                kkt = np.block([[k[np.ix_(free, free)], e], [e.T, np.zeros((m, m))]])
                linear = c[free] - sign * costs[free] - k[np.ix_(free, held)] @ w[held]
                solution, *_ = np.linalg.lstsq(kkt, np.concatenate([linear, rhs]), rcond=None)
                candidate = w.copy()
                candidate[free] = solution[: len(free)]
                candidates = [candidate]
            for candidate in candidates:
                tolerance = 1e-12 * max(1.0, np.abs(candidate).max())
                if abs(candidate.sum() - 1.0) > 1e-9:
                    continue
                if np.any(candidate < lower - tolerance) or np.any(candidate > upper + tolerance):
                    continue
                moved = candidate[free] - current[free]
                if np.any(sign * moved < -tolerance):
                    continue
                if cap is not None and np.abs(candidate - current).sum() > cap + 1e-12:
                    continue
                value = objective(candidate)
                if value < best_value:
                    best, best_value = candidate, value
    return best


def rebalance(cov: np.ndarray, mu: np.ndarray, params: dict, mean_variance: bool) -> np.ndarray | str:
    """Min variance, wᵀΣw + κ‖w − w₀‖², or mean-variance, −μᵀw + (γ/2) wᵀΣw + κ‖w − w₀‖²
    + Σcᵢ|wᵢ − w₀ᵢ|, from current weights w₀, as ½wᵀKw − cᵀw plus the costs."""
    n = cov.shape[0]
    current = np.array(params["current_weights"], dtype=float)
    kappa = params.get("turnover_penalty", 0.0)
    cap = params.get("max_turnover")
    costs = params.get("transaction_costs", 0.0)
    costs = np.full(n, costs) if np.isscalar(costs) else np.array(costs, dtype=float)
    if params.get("long_short"):
        lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    else:
        bounds = weight_bounds(params, n)
        lower, upper = bounds if bounds is not None else (np.zeros(n), np.full(n, np.inf))
        # Each cap lowered to what the other floors leave, as cutup resolves them.
        upper = np.minimum(upper, 1.0 - (lower.sum() - lower))
    if mean_variance:
        gamma = params["risk_aversion"]
        k, c = gamma * cov + 2 * kappa * np.eye(n), mu + 2 * kappa * current
    else:
        k, c = 2 * cov + 2 * kappa * np.eye(n), 2 * kappa * current
    if cap is not None:
        clipped = np.clip(current, lower, upper)
        minimum = np.abs(clipped - current).sum() + abs(1.0 - clipped.sum())
        if minimum > cap + 1e-12:
            return "InfeasibleTurnover"
    weights = kinked(k, c, current, costs, lower, upper, cap)
    cross_check_rebalance(cov, mu, current, kappa, costs, cap, lower, upper, weights, params if mean_variance else None)
    return weights


def cross_check_rebalance(cov, mu, current, kappa, costs, cap, lower, upper, weights, mean_variance) -> None:
    """Checks a rebalance against PyPortfolioOpt on cvxpy, with Clarabel at tight
    tolerances, to 1e-6. Unlike the other cross-checks it holds even on near_singular4."""
    bounds = [(None if np.isinf(l) else l, None if np.isinf(u) else u) for l, u in zip(lower, upper)]
    # Clarabel at tight tolerances: at its defaults it stops about 2e-5 short at the kinks.
    tight = {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12, "max_iter": 500}
    frontier = EfficientFrontier(mu, cov, weight_bounds=bounds, solver="CLARABEL", solver_options=tight)
    if np.any(costs > 0):
        frontier.add_objective(lambda w: cp.sum(cp.multiply(costs, cp.abs(w - current))))
    if kappa > 0:
        frontier.add_objective(lambda w: kappa * cp.sum_squares(w - current))
    if cap is not None:
        frontier.add_constraint(lambda w: cp.norm(w - current, 1) <= cap)
    if mean_variance is not None:
        frontier.max_quadratic_utility(mean_variance["risk_aversion"])
    else:
        frontier.min_volatility()
    theirs = np.array(list(frontier.weights))
    if np.abs(theirs - weights).max() > 1e-6:
        raise ValueError(f"PyPortfolioOpt disagrees: {theirs} vs {weights}")


def black_litterman(cov: np.ndarray, params: dict) -> np.ndarray:
    """Posterior expected returns. Each view is {"absolute": i}, {"relative": [long, short]}
    or {"portfolio": weights}, with a return "q" and optionally a "variance" or a
    "confidence"."""
    n = cov.shape[0]
    prior = market_implied_prior_returns(
        pd.Series(params["market_weights"]),
        params["risk_aversion"],
        pd.DataFrame(cov),
        params.get("risk_free_rate", 0.0),
    ).to_numpy()
    tau = params.get("tau", 0.05)
    views = params.get("views", [])
    if not views:
        return prior
    p = np.zeros((len(views), n))
    for row, view in zip(p, views):
        if "absolute" in view:
            row[view["absolute"]] = 1.0
        elif "relative" in view:
            row[view["relative"][0]], row[view["relative"][1]] = 1.0, -1.0
        else:
            row[:] = view["portfolio"]
    q = np.array([view["q"] for view in views])
    if any("variance" in view and "confidence" in view for view in views):
        raise ValueError("a view sets at most one of variance or confidence")
    kinds = {"variance" if "variance" in v else "confidence" if "confidence" in v else "default" for v in views}
    if kinds == {"default"}:
        omega, confidences = "default", None
    elif kinds == {"confidence"}:
        omega, confidences = "idzorek", [view["confidence"] for view in views]
    else:
        variances = []
        for row, view in zip(p, views):
            row = row.reshape(1, -1)
            if "variance" in view:
                variances.append(view["variance"])
            elif "confidence" in view:
                omega_row = BlackLittermanModel.idzorek_method([view["confidence"]], cov, prior, [view["q"]], row, tau)
                variances.append(omega_row[0, 0])
            else:
                variances.append(BlackLittermanModel.default_omega(cov, row, tau)[0, 0])
        omega, confidences = np.diag(variances), None
    model = BlackLittermanModel(
        pd.DataFrame(cov), pi=prior, P=p, Q=q, omega=omega, view_confidences=confidences, tau=tau
    )
    return model.bl_returns().to_numpy()


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
            return long_only_min_variance(adjusted, params)
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
        case "max_diversification":
            return max_diversification(adjusted)
        case "risk_parity":
            budgets = np.array(params.get("risk_budgets", np.ones(returns.shape[1])), dtype=float)
            return risk_parity(adjusted, budgets)
        case "hrp":
            return hrp(returns, params.get("linkage", "single"))
        case "herc":
            return herc(returns, params.get("linkage", "single"), params.get("clusters"))
        case "black_litterman":
            return black_litterman(adjusted, params)
        case "rebalance_min_variance":
            return rebalance(adjusted, mu, params, mean_variance=False)
        case "rebalance_mean_variance":
            return rebalance(adjusted, mu, params, mean_variance=True)
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

    # Max diversification on every dataset. The singular ones have their optimum unique only
    # with shrinkage, or have no maximum at all: rank_deficient6 has a long-only portfolio with
    # zero variance.
    for name in ["orthogonal3", "two_asset", "two_blocks5", "prices10x60", "near_singular4"]:
        out.append({"dataset": name, "method": "max_diversification", "tolerance": solver})
    for name in ["two_blocks5", "prices10x60", "rank_deficient6", "collinear2"]:
        out.append({"dataset": name, "method": "max_diversification", "params": {"shrinkage": 0.3}, "tolerance": solver})
    out.append({"dataset": "near_singular4", "method": "max_diversification", "params": {"regularization": 1e-5}, "tolerance": solver})
    out.append({"dataset": "rank_deficient6", "method": "max_diversification", "tolerance": solver})
    # Volatilities eight orders of magnitude apart, with expected returns in proportion to
    # them, so that every asset matters.
    out.append({"dataset": "scales4", "method": "max_diversification", "tolerance": solver})
    out.append({"dataset": "scales4", "method": "long_only_min_variance", "tolerance": solver})
    scaled_mu = [3.471e-04, 7.583e-07, 6.112e-10, 1.783e-12]
    for gamma in [1e6, 1e10]:
        for method in ["mean_variance", "long_only_mean_variance"]:
            out.append({"dataset": "scales4", "method": method, "params": {"risk_aversion": gamma, "expected_returns": scaled_mu}, "tolerance": solver})
    for method in ["mean_variance", "long_only_mean_variance"]:
        out.append({"dataset": "scales4", "method": method, "params": {"target_return": 6.942e-06, "expected_returns": scaled_mu}, "tolerance": solver})
    for method in ["max_sharpe", "long_only_max_sharpe"]:
        out.append({"dataset": "scales4", "method": method, "params": {"expected_returns": scaled_mu}, "tolerance": solver})

    # Weight bounds on long-only min variance and mean-variance: caps and floors that bind,
    # bounds no portfolio summing to 1 meets, and targets inside, at the end of, and beyond
    # the range of returns the bounds allow.
    for name, params in [
        ("orthogonal3", {"bounds": [0.0, 0.4]}),
        ("two_blocks5", {"bounds": [0.0, 0.3]}),
        ("prices10x60", {"bounds": [0.0, 0.15]}),
        ("prices10x60", {"bounds": [0.05, 0.3]}),
        ("near_singular4", {"lower_bounds": [0.1, 0.0, 0.1, 0.0], "upper_bounds": [0.5, 0.6, 1.0, 0.35]}),
        ("scales4", {"bounds": [0.0, 0.5]}),
        ("rank_deficient6", {"bounds": [0.0, 0.25], "shrinkage": 0.3}),
        ("orthogonal3", {"bounds": [0.0, 0.3]}),
        ("two_blocks5", {"bounds": [0.25, 1.0]}),
    ]:
        out.append({"dataset": name, "method": "long_only_min_variance", "params": params, "tolerance": solver})
    mu10 = [0.0010, 0.0004, 0.0007, 0.0002, 0.0012, 0.0005, 0.0003, 0.0008, 0.0006, 0.0009]
    # The first two assets tie, and a cap of 40% splits the weight left at the margin between
    # them at the highest return, 0.4 · 0.0012 + 0.6 · 0.0010.
    tied = [0.0010, 0.0010, 0.0007, 0.0002, 0.0012, 0.0005, 0.0003, 0.0008, 0.0006, 0.0009]
    for name, params in [
        ("two_blocks5", {"risk_aversion": 5.0, "bounds": [0.0, 0.35]}),
        ("prices10x60", {"risk_aversion": 5.0, "bounds": [0.0, 0.2]}),
        ("prices10x60", {"risk_aversion": 100.0, "bounds": [0.02, 0.25], "expected_returns": mu10}),
        # A cap of 25% allows returns from 0.00035 to 0.000975.
        ("prices10x60", {"target_return": 0.0008, "bounds": [0.0, 0.25], "expected_returns": mu10}),
        ("prices10x60", {"target_return": 0.000975, "bounds": [0.0, 0.25], "expected_returns": mu10}),
        ("prices10x60", {"target_return": 0.001, "bounds": [0.0, 0.25], "expected_returns": mu10}),
        ("prices10x60", {"target_return": 0.00108, "bounds": [0.0, 0.4], "expected_returns": tied}),
        ("two_blocks5", {"risk_aversion": 5.0, "bounds": [0.0, 0.15]}),
    ]:
        out.append({"dataset": name, "method": "long_only_mean_variance", "params": params, "tolerance": solver})

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

    # Rebalancing from current weights: turnover caps that bind and don't, a turnover penalty,
    # transaction costs, uniform and per asset, with bounds the current weights break, long-
    # short, and a cap too small to reach the bounds.
    for name, params in [
        ("orthogonal3", {"current_weights": [0.6, 0.3, 0.1], "max_turnover": 0.2}),
        ("orthogonal3", {"current_weights": [0.6, 0.3, 0.1], "max_turnover": 2.0}),
        ("orthogonal3", {"current_weights": [0.6, 0.3, 0.1], "max_turnover": 0.0}),
        ("two_asset", {"current_weights": [0.9, 0.1], "max_turnover": 0.1}),
        ("two_blocks5", {"current_weights": [0.2] * 5, "max_turnover": 0.15}),
        ("two_blocks5", {"current_weights": [0.5, 0.0, 0.0, 0.25, 0.25], "turnover_penalty": 2e-4}),
        ("two_blocks5", {"current_weights": [0.5, 0.0, 0.0, 0.25, 0.25], "turnover_penalty": 1e-4, "max_turnover": 0.3}),
        ("near_singular4", {"current_weights": [0.1, 0.2, 0.3, 0.4], "max_turnover": 0.25}),
        ("near_singular4", {"current_weights": [0.1, 0.2, 0.3, 0.4], "max_turnover": 0.25, "bounds": [0.0, 0.35]}),
        ("near_singular4", {"current_weights": [0.1, 0.2, 0.3, 0.4], "max_turnover": 0.05, "bounds": [0.0, 0.35]}),
        ("two_blocks5", {"current_weights": [0.4, 0.3, 0.1, 0.1, 0.1], "max_turnover": 0.2, "long_short": True}),
    ]:
        out.append({"dataset": name, "method": "rebalance_min_variance", "params": params, "tolerance": solver})
    mu5 = [0.0012, 0.0004, 0.0009, 0.0002, 0.0007]
    for name, params in [
        ("two_blocks5", {"risk_aversion": 50.0, "expected_returns": mu5, "current_weights": [0.2] * 5, "transaction_costs": 0.0005}),
        ("two_blocks5", {"risk_aversion": 50.0, "expected_returns": mu5, "current_weights": [0.2] * 5, "transaction_costs": [0.001, 0.0, 0.0005, 0.0002, 0.002]}),
        ("two_blocks5", {"risk_aversion": 50.0, "expected_returns": mu5, "current_weights": [0.2] * 5, "transaction_costs": 0.0002, "max_turnover": 0.3}),
        ("two_blocks5", {"risk_aversion": 50.0, "expected_returns": mu5, "current_weights": [0.2] * 5, "turnover_penalty": 1e-3, "transaction_costs": 0.0001}),
        ("two_blocks5", {"risk_aversion": 20.0, "expected_returns": mu5, "current_weights": [0.4, 0.3, 0.1, 0.1, 0.1], "transaction_costs": 0.0003, "long_short": True}),
        ("two_blocks5", {"risk_aversion": 20.0, "expected_returns": mu5, "current_weights": [0.4, 0.3, 0.1, 0.1, 0.1], "transaction_costs": 0.0003, "max_turnover": 0.4, "long_short": True}),
        ("orthogonal3", {"risk_aversion": 100.0, "expected_returns": [0.002, 0.001, 0.0], "current_weights": [0.1, 0.1, 0.8], "transaction_costs": 0.0002, "bounds": [0.0, 0.5]}),
        ("near_singular4", {"risk_aversion": 100.0, "current_weights": [0.25] * 4, "transaction_costs": 0.0001, "max_turnover": 0.1}),
    ]:
        out.append({"dataset": name, "method": "rebalance_mean_variance", "params": params, "tolerance": solver})

    # Black-Litterman expected returns: the prior alone, then each kind of view and of view
    # uncertainty, alone and mixed. Market capitalizations, not weights, on prices10x60.
    caps10 = [12.0, 3.5, 8.0, 1.2, 20.0, 4.4, 2.5, 9.1, 6.3, 5.0]
    linear = 1e-15
    for name, params in [
        ("prices10x60", {"market_weights": caps10, "risk_aversion": 2.5}),
        ("prices10x60", {"market_weights": caps10, "risk_aversion": 3.0, "risk_free_rate": 0.0001}),
        ("prices10x60", {"market_weights": caps10, "risk_aversion": 2.5, "views": [
            {"absolute": 4, "q": 0.002},
            {"relative": [0, 3], "q": 0.0005},
            {"portfolio": [0.5, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0], "q": 0.0003},
        ]}),
        ("prices10x60", {"market_weights": caps10, "risk_aversion": 2.5, "tau": 0.2, "views": [
            {"absolute": 4, "q": 0.002, "confidence": 0.8},
            {"relative": [0, 3], "q": 0.0005, "confidence": 0.3},
        ]}),
        ("prices10x60", {"market_weights": caps10, "risk_aversion": 2.5, "views": [
            {"absolute": 4, "q": 0.002, "confidence": 0.05},
            {"relative": [0, 3], "q": 0.0005, "confidence": 0.95},
        ]}),
        ("prices10x60", {"market_weights": caps10, "risk_aversion": 2.5, "shrinkage": 0.3, "views": [
            {"absolute": 4, "q": 0.002, "variance": 1e-6},
            {"relative": [0, 3], "q": 0.0005, "confidence": 0.6},
            {"absolute": 9, "q": -0.001},
            {"relative": [8, 1], "q": 0.0002, "confidence": 1.0},
        ]}),
        ("two_blocks5", {"market_weights": [0.3, 0.2, 0.2, 0.15, 0.15], "risk_aversion": 2.5, "views": [
            {"relative": [0, 3], "q": 0.001},
        ]}),
        ("near_singular4", {"market_weights": [0.25, 0.25, 0.25, 0.25], "risk_aversion": 2.0, "views": [
            {"absolute": 0, "q": 0.001, "confidence": 0.5},
            {"relative": [1, 2], "q": 0.0005, "variance": 0.0},
        ]}),
        ("orthogonal3", {"market_weights": [0.5, 0.3, 0.2], "risk_aversion": 2.5, "regularization": 1e-5, "views": [
            {"absolute": 2, "q": 0.01, "variance": 0.0},
        ]}),
    ]:
        out.append({"dataset": name, "method": "black_litterman", "params": params, "tolerance": linear})
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
        if case["method"] in ("rebalance_min_variance", "rebalance_mean_variance", "min_variance", "long_only_min_variance", "long_only_mean_variance", "long_only_max_sharpe", "max_diversification", "mean_variance", "max_sharpe"):
            # A numerically singular covariance has its null space truncated, which
            # every solver does alike. Short of that, any solver loses about cond(Σ) · ε
            # of accuracy, so ill-conditioned cases get a proportionally looser tolerance.
            # Every solve is in correlation units, so its accuracy depends on the
            # correlation matrix's conditioning, not on each asset's units.
            condition = float(np.linalg.cond(correlation(adjusted)))
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
