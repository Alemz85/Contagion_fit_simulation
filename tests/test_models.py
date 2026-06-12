"""Analytic edge-case tests for the contagion models (DESIGN.md section 6).

These pin the models to hand-computable answers on small graphs:
    p = 0      -> IC activates only the seeds
    p = 1      -> IC activates the seed's whole connected component
    phi = 0    -> threshold activates everything
    phi > 1    -> threshold activates only the seeds
"""

import networkx as nx
import numpy as np
import pytest

from contagion_fit.models import ComplexContagion, SimpleContagion


def _relabelled(graph: nx.Graph) -> nx.Graph:
    """Models assume integer labels 0..n-1."""
    return nx.convert_node_labels_to_integers(graph)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- Independent Cascade -------------------------------------------------
def test_ic_p_zero_activates_only_seeds(rng):
    g = _relabelled(nx.path_graph(10))
    result = SimpleContagion(p=0.0).run(g, {0}, rng)
    assert result == {0}


def test_ic_p_one_fills_connected_component(rng):
    g = _relabelled(nx.path_graph(10))
    result = SimpleContagion(p=1.0).run(g, {0}, rng)
    assert result == set(range(10))


def test_ic_p_one_stays_in_component(rng):
    # Two disjoint triangles: seeding one must not reach the other.
    g = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    result = SimpleContagion(p=1.0).run(g, {0}, rng)
    assert result == {0, 1, 2}


def test_ic_star_full_spread_on_p_one(rng):
    g = _relabelled(nx.star_graph(7))  # node 0 is the hub
    result = SimpleContagion(p=1.0).run(g, {0}, rng)
    assert result == set(range(8))


# --- Complex (threshold) contagion --------------------------------------
def test_threshold_phi_zero_activates_everything(rng):
    g = _relabelled(nx.path_graph(10))
    result = ComplexContagion(phi=0.0).run(g, {0}, rng)
    assert result == set(range(10))


def test_threshold_phi_above_one_keeps_only_seeds(rng):
    g = _relabelled(nx.complete_graph(6))
    result = ComplexContagion(phi=1.5).run(g, {0}, rng)
    assert result == {0}


def test_threshold_complete_graph_cascades_with_low_phi(rng):
    # In K6 every non-seed has 1/5 = 0.2 of its neighbours active initially,
    # so phi = 0.2 lets the cascade complete.
    g = _relabelled(nx.complete_graph(6))
    result = ComplexContagion(phi=0.2).run(g, {0}, rng)
    assert result == set(range(6))


def test_threshold_high_phi_blocks_on_path(rng):
    # On a path each interior node has degree 2; with phi = 0.6 it needs
    # ceil(0.6*2) = 2 active neighbours, but the seed only provides one side,
    # so nothing past the immediate neighbour can ever ignite.
    g = _relabelled(nx.path_graph(10))
    result = ComplexContagion(phi=0.6).run(g, {0}, rng)
    assert result == {0}


def test_models_do_not_mutate_seeds(rng):
    g = _relabelled(nx.path_graph(5))
    seeds = {0}
    SimpleContagion(p=0.5).run(g, seeds, rng)
    assert seeds == {0}


def test_run_returns_superset_of_seeds(rng):
    g = _relabelled(nx.path_graph(20))
    seeds = {3, 7}
    result = SimpleContagion(p=0.3).run(g, seeds, rng)
    assert seeds <= result
