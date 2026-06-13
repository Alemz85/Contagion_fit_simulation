"""Tests for the Phase 4 seed-budget / cost extension.

These pin the new machinery to checkable facts:
    - the degree cost model rises with degree,
    - budgeted selection never overspends and ``cost_alpha = 0`` reduces to a
      plain k-seed budget,
    - the fixed-seed Monte-Carlo path reproduces the star-graph analytic mean,
    - greedy influence maximisation is at least as good as degree seeding, which
      is at least as good as random seeding (the headline ordering).
"""

import networkx as nx
import numpy as np
import pytest

from contagion_fit.config import Config
from contagion_fit.experiments import (
    greedy_influence_max,
    seed_budget_experiment,
)
from contagion_fit.models import SimpleContagion
from contagion_fit.network import centrality_scores, degree_cost
from contagion_fit.simulate import (
    budget_seed_order,
    select_seeds_under_budget,
)


def _relabelled(graph: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(graph)


# --- cost model ----------------------------------------------------------
def test_degree_cost_rises_with_degree():
    g = _relabelled(nx.star_graph(20))  # hub deg 20, leaves deg 1
    cost = degree_cost(g, alpha=1.0)
    hub = int(np.argmax([d for _, d in g.degree()]))
    assert cost[hub] == cost.max()
    assert cost.min() >= 1.0  # baseline price floor
    assert cost[hub] > cost.min()  # a hub is strictly pricier than a leaf


def test_cost_alpha_zero_is_flat():
    g = _relabelled(nx.barabasi_albert_graph(100, 3, seed=1))
    cost = degree_cost(g, alpha=0.0)
    assert np.allclose(cost, 1.0)


# --- budgeted selection --------------------------------------------------
def test_budget_never_overspends():
    g = _relabelled(nx.barabasi_albert_graph(300, 3, seed=2))
    cost = degree_cost(g, alpha=1.0)
    rng = np.random.default_rng(0)
    for strat in ("random", "hub", "pagerank"):
        scores = None if strat == "random" else centrality_scores(g, strat, rng)
        order = budget_seed_order(g.number_of_nodes(), strat, rng, scores=scores)
        seeds = select_seeds_under_budget(cost, budget=15.0, order=order)
        spent = cost[np.fromiter(seeds, dtype=np.int64)].sum()
        assert spent <= 15.0 + 1e-9
        assert len(seeds) >= 1


def test_alpha_zero_budget_is_k_seeds():
    # With unit costs a budget of B buys exactly floor(B) seeds.
    g = _relabelled(nx.barabasi_albert_graph(200, 2, seed=3))
    cost = degree_cost(g, alpha=0.0)
    order = budget_seed_order(g.number_of_nodes(), "hub", np.random.default_rng(0),
                              scores=centrality_scores(g, "hub"))
    seeds = select_seeds_under_budget(cost, budget=7.0, order=order)
    assert len(seeds) == 7


# --- fixed-seed Monte-Carlo path matches the star analytic ---------------
def test_fixed_seed_reach_matches_star_analytic():
    # Seed the hub of a k-leaf star: one IC step activates each leaf with prob p,
    # so E[reach] = 1 + k*p. This validates the new fixed-seed MC driver.
    k, p = 30, 0.3
    config = Config(n_nodes=200, n_workers=1, seed=7)
    # Build a star directly and reuse the in-process serial path.
    g = _relabelled(nx.star_graph(k))
    from contagion_fit.simulate import reach_fixed_seeds_on_graph
    child = [config.child_seed(i) for i in range(20000)]
    reach = reach_fixed_seeds_on_graph(g, SimpleContagion(p=p), {0}, child)
    assert reach.mean() == pytest.approx(1 + k * p, abs=0.15)


# --- greedy >= degree >= random -----------------------------------------
def _reach_on_graph(g, model, seeds, config, n_runs, stream):
    from contagion_fit.simulate import reach_fixed_seeds_on_graph
    child = [config.child_seed(stream + i) for i in range(n_runs)]
    return reach_fixed_seeds_on_graph(g, model, set(seeds), child).mean()


def test_greedy_ge_degree_ge_random_on_disjoint_stars():
    # Two disjoint stars (hub degrees 6 and 4) with deterministic IC (p=1).
    # Reach = size of the components the seeds touch.
    #   degree picks the two hubs (one per star) -> reaches everything (12),
    #   greedy also picks one hub per star        -> 12,
    #   random can waste both seeds in one star    -> expected reach < 12.
    # All reach is evaluated directly on this hand-built graph (the config-driven
    # parallel drivers rebuild a substrate, so they are not used here).
    s1 = nx.star_graph(6)                       # 7 nodes, hub deg 6
    s2 = nx.star_graph(4)                        # 5 nodes, hub deg 4
    g = _relabelled(nx.disjoint_union(s1, s2))   # 12 nodes total
    total = g.number_of_nodes()
    model = SimpleContagion(p=1.0)
    cost = degree_cost(g, alpha=0.0)             # unit cost -> 2-seed budget
    config = Config(n_nodes=total, n_workers=1, seed=11)

    # degree seeding: top-2 by degree
    scores = centrality_scores(g, "degree")
    deg_order = budget_seed_order(total, "hub", config.rng(), scores=scores)
    deg_seeds = select_seeds_under_budget(cost, budget=2.0, order=deg_order)
    deg_reach = _reach_on_graph(g, model, deg_seeds, config, 200, 1000)

    # greedy seeding (evaluates marginal reach on g itself)
    g_seeds, _ = greedy_influence_max(
        config, model, g, cost, budget=2.0, eval_runs=40, pool_size=12,
    )
    greedy_reach = _reach_on_graph(g, model, g_seeds, config, 200, 2000)

    # random under the same budget: redraw two cheap seeds each run.
    rng = np.random.default_rng(99)
    rnd_sizes = []
    for _ in range(600):
        order = budget_seed_order(total, "random", rng)
        rnd_seeds = select_seeds_under_budget(cost, budget=2.0, order=order)
        rnd_sizes.append(_reach_on_graph(g, model, rnd_seeds, config, 1, 3000))
    rnd_reach = float(np.mean(rnd_sizes))

    assert deg_reach == pytest.approx(total)        # both hubs -> whole graph
    assert greedy_reach >= deg_reach - 1e-9
    assert deg_reach >= rnd_reach - 1e-9
    assert rnd_reach < total                          # random sometimes wastes a seed


# --- end-to-end smoke of the experiment ---------------------------------
def test_seed_budget_experiment_runs_small():
    config = Config(n_nodes=400, n_runs=60, n_workers=1, seed=5,
                    seed_budget=8.0, cost_alpha=1.0,
                    seed_strategies=("random", "hub", "greedy"),
                    greedy_eval_runs=20, greedy_pool=10)
    result = seed_budget_experiment(config, SimpleContagion(p=0.1))
    assert {r.strategy for r in result.results} == {"random", "hub", "greedy"}
    for r in result.results:
        assert r.reach.size == 60
        assert r.seed_set_size >= 1
        assert r.spent_cost <= config.seed_budget + 1e-9
    # the experiment must name a reach winner and a reach-per-cost winner
    assert result.best_by_reach().strategy in {"random", "hub", "greedy"}
    assert result.best_by_reach_per_cost().strategy in {"random", "hub", "greedy"}
