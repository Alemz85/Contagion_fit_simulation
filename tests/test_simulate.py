"""Reproducibility and analytic-expectation tests for the Monte-Carlo engine.

Covers:
    - a fixed rng gives bit-identical results across repeated runs
    - the star-graph IC mean size matches the analytic value 1 + k*p
    - seeding strategies select the expected nodes
"""

import networkx as nx
import numpy as np
import pytest

from contagion_fit.models import SimpleContagion
from contagion_fit.simulate import cascade_size_distribution, choose_seeds


def _relabelled(graph: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(graph)


def test_fixed_seed_is_reproducible():
    g = _relabelled(nx.barabasi_albert_graph(200, 2, seed=1))
    model = SimpleContagion(p=0.05)
    a = cascade_size_distribution(g, model, n_runs=50, rng=np.random.default_rng(42))
    b = cascade_size_distribution(g, model, n_runs=50, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    g = _relabelled(nx.barabasi_albert_graph(200, 2, seed=1))
    model = SimpleContagion(p=0.05)
    a = cascade_size_distribution(g, model, n_runs=50, rng=np.random.default_rng(1))
    b = cascade_size_distribution(g, model, n_runs=50, rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_star_ic_mean_matches_analytic():
    # Seeding the hub of a k-leaf star and running one IC step: each leaf
    # activates independently with probability p, so E[size] = 1 + k*p.
    k = 30
    p = 0.3
    g = _relabelled(nx.star_graph(k))  # hub is node 0, k leaves
    model = SimpleContagion(p=p)
    # Always seed the hub by ranking on degree (hub has the highest degree).
    sizes = cascade_size_distribution(
        g, model, n_runs=20000, seed_strategy="hub",
        rng=np.random.default_rng(7),
    )
    expected = 1 + k * p
    assert np.mean(sizes) == pytest.approx(expected, abs=0.15)


def test_hub_seed_selects_max_degree_node():
    g = _relabelled(nx.star_graph(5))  # node 0 is the hub (degree 5)
    seeds = choose_seeds(g, "hub", 1, np.random.default_rng(0))
    assert seeds == {0}


def test_random_seed_count_respected():
    g = _relabelled(nx.path_graph(50))
    seeds = choose_seeds(g, "random", 5, np.random.default_rng(0))
    assert len(seeds) == 5


def test_distribution_length_matches_n_runs():
    g = _relabelled(nx.path_graph(20))
    sizes = cascade_size_distribution(
        g, SimpleContagion(p=0.1), n_runs=37, rng=np.random.default_rng(0)
    )
    assert sizes.shape == (37,)
