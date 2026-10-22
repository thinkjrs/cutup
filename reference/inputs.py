"""Builds the input datasets for cutup's reference tests and freezes them to JSON.

Run this only when adding or changing a dataset:

    uv run --project reference python reference/inputs.py

The output, tests/fixtures/datasets.json, is committed. Every value is written as a
literal number, so the Rust tests and the Python reference read exactly the same
inputs and nothing depends on a random number generator matching across languages
or versions.
"""

import numpy as np

from fixtures import FIXTURES, write_json

OUTPUT = FIXTURES / "datasets.json"


def orthogonal3() -> dict:
    # Mutually orthogonal, zero-mean columns: a diagonal covariance with closed-form
    # answers, and equal correlation distances, which exercises HRP's tie breaking.
    s1, s2, s3 = 0.01, 0.02, 0.04
    returns = [
        [s1, s2, s3],
        [-s1, s2, -s3],
        [s1, -s2, -s3],
        [-s1, -s2, s3],
    ]
    return {
        "kind": "returns",
        "description": "Three uncorrelated assets with a diagonal covariance.",
        "data": returns,
    }


def two_asset() -> dict:
    returns = [
        [0.012, 0.004],
        [-0.008, 0.002],
        [0.015, -0.006],
        [-0.004, 0.009],
        [0.007, 0.001],
        [-0.011, -0.003],
        [0.003, 0.006],
        [0.009, -0.002],
    ]
    return {
        "kind": "returns",
        "description": "Two correlated assets, eight periods.",
        "data": returns,
    }


def two_blocks5(rng: np.random.Generator) -> dict:
    # Assets 0, 2 and 4 follow one factor and assets 1 and 3 another, interleaved so
    # that HRP has to find the blocks rather than read them off the column order.
    periods = 30
    factors = rng.normal(0.0, 0.01, (periods, 2))
    noise = rng.normal(0.0, 0.003, (periods, 5))
    returns = np.column_stack([factors[:, j % 2] for j in range(5)]) + noise
    return {
        "kind": "returns",
        "description": "Five assets in two interleaved factor blocks, 30 periods.",
        "data": np.round(returns, 6).tolist(),
    }


def prices10x60(rng: np.random.Generator) -> dict:
    # Ten assets in three sectors, 61 days of prices (60 returns), so from_prices is
    # exercised end to end on realistic magnitudes.
    days, assets = 60, 10
    sector = np.arange(assets) % 3
    market = rng.normal(0.0004, 0.008, days)
    sectors = rng.normal(0.0, 0.006, (days, 3))
    idiosyncratic = rng.normal(0.0, 0.01, (days, assets))
    drift = rng.uniform(-0.0002, 0.0008, assets)
    returns = market[:, None] + sectors[:, sector] + idiosyncratic + drift
    start = rng.uniform(20.0, 400.0, assets)
    prices = start * np.vstack([np.ones(assets), np.cumprod(1.0 + returns, axis=0)])
    return {
        "kind": "prices",
        "description": "Ten assets in three sectors, 61 daily prices.",
        "data": np.round(prices, 2).tolist(),
    }


def near_singular4(rng: np.random.Generator) -> dict:
    # Asset 3 is asset 2 plus tiny noise: a nearly collinear pair and an
    # ill-conditioned covariance, for testing shrinkage and regularization.
    periods = 20
    base = rng.normal(0.0005, 0.012, (periods, 3))
    twin = base[:, 2] + rng.normal(0.0, 1e-5, periods)
    returns = np.column_stack([base, twin])
    return {
        "kind": "returns",
        "description": "Four assets where asset 3 nearly duplicates asset 2, 20 periods.",
        "data": np.round(returns, 8).tolist(),
    }


def rank_deficient6(rng: np.random.Generator) -> dict:
    # Four periods for six assets: the sample covariance has rank 3.
    returns = rng.normal(0.001, 0.015, (4, 6))
    return {
        "kind": "returns",
        "description": "Six assets with four periods, so the covariance is singular.",
        "data": np.round(returns, 6).tolist(),
    }


def main() -> None:
    rng = np.random.default_rng(20261022)
    datasets = {
        "orthogonal3": orthogonal3(),
        "two_asset": two_asset(),
        "two_blocks5": two_blocks5(rng),
        "prices10x60": prices10x60(rng),
        "near_singular4": near_singular4(rng),
        "rank_deficient6": rank_deficient6(rng),
    }
    write_json(OUTPUT, datasets)
    print(f"wrote {len(datasets)} datasets to {OUTPUT}")


if __name__ == "__main__":
    main()
