"""Computes reference outcomes for cutup's linear program solver.

    uv run --project reference python reference/lp.py           # write
    uv run --project reference python reference/lp.py --check   # compare

Writes tests/fixtures/lp.json, which the unit tests in src/lp.rs check the solver against:
random problems min cᵀx subject to Ax = b and l ≤ x ≤ u, some infeasible and some unbounded,
solved by scipy.optimize.linprog with HiGHS at feasibility tolerances of 1e-10, tighter than
its default 1e-7, so that the 1e-9 comparisons hold. Only the outcome and the optimal objective
are compared, since a degenerate problem can have many optimal x. Infinite bounds are written as
null, which JSON can hold. --check also compares the committed inputs exactly.
"""

import json
import sys

import numpy as np
from scipy.optimize import linprog

from fixtures import FIXTURES, write_json

OUTPUT = FIXTURES / "lp.json"
TIGHT = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def bound(value: float) -> float | None:
    return None if np.isinf(value) else float(value)


def case(rng: np.random.Generator, kind: str) -> dict:
    m = int(rng.integers(1, 6))
    n = int(rng.integers(m + 1, 10))
    a = np.round(rng.normal(size=(m, n)), 3)
    # Roughly a third of the variables are degenerate-prone integers.
    a[:, : n // 3] = np.round(a[:, : n // 3])
    lower = np.where(rng.random(n) < 0.2, -np.inf, np.round(rng.uniform(-1, 1, n), 2))
    upper = np.where(rng.random(n) < 0.4, np.inf, lower + np.round(rng.uniform(0, 2, n), 2))
    upper = np.where(np.isinf(lower) & np.isfinite(upper), np.round(rng.uniform(-1, 1, n), 2), upper)
    # A point within the bounds, so that b = Ax₀ is feasible.
    x0 = np.where(np.isfinite(lower), lower, np.where(np.isfinite(upper), upper - 1.0, 0.0))
    x0 = np.where(np.isfinite(upper), (x0 + upper) / 2, x0 + 0.5)
    b = np.round(a @ x0, 6)
    c = np.round(rng.normal(size=n), 3)
    if kind == "bounded":
        # Costs that can't fall without limit along an unbounded variable.
        c = np.where(np.isinf(upper), np.abs(c), c)
        c = np.where(np.isinf(lower), -np.abs(c), c)
        c = np.where(np.isinf(lower) & np.isinf(upper), 0.0, c)
    # "unadjusted" costs are left as drawn, so a variable without a bound in the direction its
    # cost favors makes the problem unbounded.
    elif kind == "infeasible":
        # Every variable boxed, and a row that can't reach its right-hand side.
        lower, upper = np.zeros(n), np.ones(n)
        a[0] = np.abs(a[0])
        b[0] = a[0].sum() + 1.0
    result = linprog(c, A_eq=a, b_eq=b, bounds=list(zip(np.where(np.isinf(lower), None, lower), np.where(np.isinf(upper), None, upper))), method="highs", options=TIGHT)
    outcome = {0: "optimal", 2: "infeasible", 3: "unbounded"}[result.status]
    out = {
        "c": c.tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "lower": [bound(v) for v in lower],
        "upper": [bound(v) for v in upper],
        "outcome": outcome,
    }
    if outcome == "optimal":
        out["objective"] = float(result.fun)
    return out


def cases() -> list[dict]:
    rng = np.random.default_rng(20261028)
    kinds = ["bounded"] * 40 + ["unadjusted"] * 20 + ["infeasible"] * 10
    return [case(rng, kind) for kind in kinds]


def main() -> None:
    fresh = {"cases": cases()}
    if "--check" in sys.argv[1:]:
        committed = json.loads(OUTPUT.read_text())
        inputs = ("c", "a", "b", "lower", "upper")
        problems = [
            i
            for i, (old, new) in enumerate(zip(committed["cases"], fresh["cases"]))
            if any(old[key] != new[key] for key in inputs)
            or old["outcome"] != new["outcome"]
            or abs(old.get("objective", 0.0) - new.get("objective", 0.0)) > 1e-9 * max(1.0, abs(new.get("objective", 0.0)))
        ]
        if len(committed["cases"]) != len(fresh["cases"]):
            problems.append("count")
        print(f"{len(problems)} of {len(fresh['cases'])} cases drifted from {OUTPUT}")
        sys.exit(1 if problems else 0)
    write_json(OUTPUT, fresh)
    counts = {k: sum(c["outcome"] == k for c in fresh["cases"]) for k in ["optimal", "infeasible", "unbounded"]}
    print(f"wrote {len(fresh['cases'])} cases {counts} to {OUTPUT}")


if __name__ == "__main__":
    main()
