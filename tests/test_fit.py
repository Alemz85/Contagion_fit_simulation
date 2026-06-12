"""Fitting tests, including the parameter-recovery test (DESIGN.md section 6).

The recovery test is the methodological keystone: we generate synthetic data
from the IC model at a known ``p``, feed it to the grid search, and check the
true ``p`` is recovered. If this passes, the fitting machinery is trustworthy.
"""

import numpy as np
import pytest

from contagion_fit.config import Config
from contagion_fit.fit import (
    decide_winner,
    grid_search,
    ks_distance,
)
from contagion_fit.models import SimpleContagion
from contagion_fit.simulate import simulate_parallel


@pytest.fixture
def small_config():
    # Single worker keeps the test fast and avoids process-pool overhead.
    return Config(n_nodes=1500, n_runs=300, n_workers=1, seed=123)


# --- KS distance properties ---------------------------------------------
def test_ks_identical_samples_is_zero():
    x = np.array([1, 2, 3, 4, 5, 10, 100])
    assert ks_distance(x, x) == pytest.approx(0.0)


def test_ks_separated_samples_is_large():
    small = np.array([1, 2, 3, 4, 5])
    large = np.array([100, 200, 300, 400, 500])
    assert ks_distance(small, large) == pytest.approx(1.0)


def test_ks_empty_is_inf():
    assert ks_distance(np.array([]), np.array([1, 2, 3])) == float("inf")


# --- winner decision -----------------------------------------------------
def test_decide_winner_simple():
    winner, margin = decide_winner(0.04, 0.20)
    assert winner == "simple"
    assert margin == pytest.approx(0.16)


def test_decide_winner_complex():
    winner, _ = decide_winner(0.30, 0.05)
    assert winner == "complex"


def test_decide_winner_tie():
    winner, _ = decide_winner(0.10, 0.105)
    assert winner == "tie"


# --- parameter recovery (the keystone) ----------------------------------
def test_parameter_recovery(small_config):
    p_true = 0.05
    observed = simulate_parallel(
        small_config, SimpleContagion(p=p_true), stream_offset=999_000
    )
    result = grid_search(
        small_config, SimpleContagion, {"p": small_config.ic_p_grid}, observed
    )
    # The recovered p should be the true value, or at worst an adjacent grid
    # point (KS on a finite sample can tie between neighbours).
    grid = list(small_config.ic_p_grid)
    true_idx = grid.index(p_true)
    recovered_idx = grid.index(result.best_params["p"])
    assert abs(recovered_idx - true_idx) <= 1


def test_grid_search_table_is_complete(small_config):
    observed = simulate_parallel(
        small_config, SimpleContagion(p=0.03), stream_offset=500_000
    )
    result = grid_search(
        small_config, SimpleContagion, {"p": small_config.ic_p_grid}, observed
    )
    assert len(result.table) == len(small_config.ic_p_grid)
    assert result.best_distance == min(c.distance for c in result.table)
