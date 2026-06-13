"""Seeding strategy and the hub/random flip boundary.

This is the direct confrontation between the influencer hypothesis (target the
hubs) and the Watts-Dodds view (a critical mass of easily-influenced ordinary
people, reached just as well by random seeding).

``seeding_experiment``
    For a fixed model, compare the reach distribution of hub seeding vs random
    seeding. Produces F4.

``flip_boundary``
    Sweep a (p, phi) plane and, at each point, measure the hub advantage
    ``mean(hub_reach) - mean(random_reach)`` for a threshold model. Where this
    changes sign, the better seeding strategy flips. Produces F5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from contagion_fit.config import Config
from contagion_fit.models import ComplexContagion, ContagionModel
from contagion_fit.network import (
    build_network,
    centrality_scores,
    degree_cost,
    degree_ranked_nodes,
)
from contagion_fit.simulate import (
    budget_seed_order,
    reach_fixed_seeds,
    reach_fixed_seeds_on_graph,
    reach_random_budget,
    select_seeds_under_budget,
    simulate_parallel,
)


@dataclass(frozen=True)
class SeedingResult:
    """Reach distributions under the two seeding strategies."""

    model_label: str
    hub: np.ndarray
    random: np.ndarray

    @property
    def hub_advantage(self) -> float:
        return float(self.hub.mean() - self.random.mean())

    def to_dict(self) -> dict:
        return {
            "model_label": self.model_label,
            "hub_mean": float(self.hub.mean()),
            "hub_std": float(self.hub.std()),
            "random_mean": float(self.random.mean()),
            "random_std": float(self.random.std()),
            "hub_advantage": self.hub_advantage,
        }


def seeding_experiment(
    config: Config,
    model: ContagionModel,
    n_runs: int | None = None,
    seed_count: int = 1,
) -> SeedingResult:
    """Compare hub vs random seeding reach for ``model``.

    The two strategies use disjoint random substreams so their comparison is
    fair and reproducible.
    """
    n_runs = config.n_runs if n_runs is None else n_runs
    hub = simulate_parallel(
        config, model, n_runs=n_runs, seed_strategy="hub",
        seed_count=seed_count, stream_offset=0,
    )
    rnd = simulate_parallel(
        config, model, n_runs=n_runs, seed_strategy="random",
        seed_count=seed_count, stream_offset=n_runs,
    )
    return SeedingResult(model_label=model.label, hub=hub, random=rnd)


@dataclass(frozen=True)
class FlipBoundaryResult:
    """Hub-advantage surface over a (p, phi) grid."""

    p_range: tuple[float, ...]
    phi_range: tuple[float, ...]
    advantage: np.ndarray  # shape (len(phi_range), len(p_range))

    def to_dict(self) -> dict:
        return {
            "p_range": list(self.p_range),
            "phi_range": list(self.phi_range),
            "advantage": self.advantage.tolist(),
        }


def flip_boundary(
    config: Config,
    p_range: Sequence[float] | None = None,
    phi_range: Sequence[float] | None = None,
    n_runs: int | None = None,
) -> FlipBoundaryResult:
    """Compute hub advantage across the (p, phi) plane for the threshold model.

    Returns a matrix indexed ``[phi, p]``; positive entries mean hub seeding
    reaches more on average, negative entries mean random seeding wins.
    """
    p_range = tuple(config.flip_p_range if p_range is None else p_range)
    phi_range = tuple(config.flip_phi_range if phi_range is None else phi_range)
    n_runs = config.flip_n_runs if n_runs is None else n_runs

    advantage = np.empty((len(phi_range), len(p_range)), dtype=np.float64)
    stream = 0
    for i, phi in enumerate(phi_range):
        for j, p in enumerate(p_range):
            model = ComplexContagion(phi=phi, p=p)
            hub = simulate_parallel(
                config, model, n_runs=n_runs, seed_strategy="hub",
                stream_offset=stream,
            )
            stream += n_runs
            rnd = simulate_parallel(
                config, model, n_runs=n_runs, seed_strategy="random",
                stream_offset=stream,
            )
            stream += n_runs
            advantage[i, j] = float(hub.mean() - rnd.mean())
    return FlipBoundaryResult(p_range=p_range, phi_range=phi_range, advantage=advantage)


# ==========================================================================
# Seed-budget experiment (Phase 4): the fair Watts-Dodds test.
#
# The original seeding experiment compares a single hub against a single random
# node, so a hub can never lose on mean reach. The real Watts-Dodds question is
# whether, under a *fixed budget*, a distributed crowd of cheap ordinary
# accounts out-reaches one expensive celebrity. We therefore:
#   1. allow a multi-seed budget (not one seed),
#   2. add stronger targeting strategies (pagerank, betweenness) and greedy
#      influence maximisation (Kempe-Kleinberg-Tardos), and
#   3. price seeds by degree, so the metric is reach-PER-COST.
# ==========================================================================

# Stream bases chosen well above the offsets used elsewhere (grid search uses
# idx*n_runs, F2 re-sim 10/20M, flip walks from 0) so budget runs never reuse
# another component's substream.
_BUDGET_STREAM_BASE = 90_000_000
_GREEDY_STREAM_BASE = 80_000_000


@dataclass(frozen=True)
class BudgetStrategyResult:
    """Reach distribution for one seeding strategy under a fixed budget."""

    strategy: str
    reach: np.ndarray          # per-run final cascade sizes
    seed_set_size: int         # how many seeds the budget bought
    spent_cost: float          # total cost of the seed set

    @property
    def mean_reach(self) -> float:
        return float(self.reach.mean())

    @property
    def sem(self) -> float:
        """Standard error of the mean reach (Monte-Carlo error bar)."""
        n = self.reach.size
        return float(self.reach.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    @property
    def reach_per_cost(self) -> float:
        return self.mean_reach / self.spent_cost if self.spent_cost > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed_set_size": self.seed_set_size,
            "spent_cost": self.spent_cost,
            "mean_reach": self.mean_reach,
            "sem": self.sem,
            "reach_per_cost": self.reach_per_cost,
        }


@dataclass(frozen=True)
class BudgetExperimentResult:
    """All strategies compared at one budget for one model."""

    model_label: str
    budget: float
    cost_alpha: float
    results: list[BudgetStrategyResult]

    def best_by_reach(self) -> BudgetStrategyResult:
        return max(self.results, key=lambda r: r.mean_reach)

    def best_by_reach_per_cost(self) -> BudgetStrategyResult:
        return max(self.results, key=lambda r: r.reach_per_cost)

    def to_dict(self) -> dict:
        return {
            "model_label": self.model_label,
            "budget": self.budget,
            "cost_alpha": self.cost_alpha,
            "best_by_reach": self.best_by_reach().strategy,
            "best_by_reach_per_cost": self.best_by_reach_per_cost().strategy,
            "results": [r.to_dict() for r in self.results],
        }


def greedy_influence_max(
    config: Config,
    model: ContagionModel,
    graph,
    cost: np.ndarray,
    *,
    budget: float,
    eval_runs: int | None = None,
    pool_size: int | None = None,
    stream_offset: int = _GREEDY_STREAM_BASE,
) -> tuple[set[int], list[dict]]:
    """Budgeted greedy influence maximisation (Kempe-Kleinberg-Tardos).

    Iteratively add the candidate with the largest *marginal expected reach per
    unit cost*, evaluated with the simulator itself, until the budget is spent.
    Evaluating every node each round is intractable on large substrates, so the
    candidate pool is the top ``pool_size`` nodes by degree (the expensive
    "celebrity shortlist") *plus* a random sample of ``pool_size`` other nodes,
    so greedy can also discover a cheap-distributed seed set rather than being
    structurally confined to hubs.

    Returns the chosen seed set and a step-by-step trace. Marginal reach is
    estimated on a prebuilt ``graph`` over ``eval_runs`` runs whose seeds are
    distinct child substreams, so the search is fully reproducible.
    """
    eval_runs = config.greedy_eval_runs if eval_runs is None else eval_runs
    pool_size = config.greedy_pool if pool_size is None else pool_size

    ranked = [int(v) for v in degree_ranked_nodes(graph)[:pool_size]]
    rest = np.setdiff1d(np.arange(graph.number_of_nodes()), np.asarray(ranked))
    rng = config.rng(stream_offset)
    extra = rng.choice(rest, size=min(pool_size, rest.size), replace=False)
    candidates = ranked + [int(v) for v in extra]
    chosen: set[int] = set()
    spent = 0.0
    trace: list[dict] = []
    stream = stream_offset

    def mean_reach(seed_set: set[int], stream_at: int) -> float:
        if not seed_set:
            return 0.0
        child = [config.child_seed(stream_at + i) for i in range(eval_runs)]
        return float(reach_fixed_seeds_on_graph(graph, model, seed_set, child).mean())

    base = 0.0  # expected reach of the current chosen set
    while True:
        best_node = None
        best_density = 0.0
        best_gain = 0.0
        for v in candidates:
            if v in chosen or spent + float(cost[v]) > budget + 1e-9:
                continue
            gain = mean_reach(chosen | {v}, stream) - base
            stream += eval_runs
            density = gain / float(cost[v])
            if best_node is None or density > best_density:
                best_node, best_density, best_gain = v, density, gain
        if best_node is None:
            break
        chosen.add(best_node)
        spent += float(cost[best_node])
        base = mean_reach(chosen, stream)
        stream += eval_runs
        trace.append({
            "added": best_node,
            "cost": float(cost[best_node]),
            "marginal_gain": best_gain,
            "spent": spent,
            "reach": base,
        })
    return chosen, trace


def seed_budget_experiment(
    config: Config,
    model: ContagionModel,
    *,
    budget: float | None = None,
    cost_alpha: float | None = None,
    strategies: Sequence[str] | None = None,
    n_runs: int | None = None,
) -> BudgetExperimentResult:
    """Compare seeding strategies at a fixed budget on equal footing.

    Deterministic strategies (hub / pagerank / betweenness / greedy) pick one
    seed set up front and are evaluated as a fixed set; ``random`` redraws cheap
    accounts each run. All share the same budget and cost model, so the reach
    (and reach-per-cost) numbers are directly comparable. Produces F8.
    """
    budget = config.seed_budget if budget is None else budget
    cost_alpha = config.cost_alpha if cost_alpha is None else cost_alpha
    strategies = config.seed_strategies if strategies is None else strategies
    n_runs = config.n_runs if n_runs is None else n_runs

    graph = build_network(config)  # deterministic; matches the workers' graph
    cost = degree_cost(graph, cost_alpha)
    n_nodes = graph.number_of_nodes()

    results: list[BudgetStrategyResult] = []
    stream = _BUDGET_STREAM_BASE
    for strat in strategies:
        if strat == "random":
            reach = reach_random_budget(
                config, model, budget, cost_alpha=cost_alpha,
                n_runs=n_runs, stream_offset=stream,
            )
            stream += n_runs
            # A representative draw, only to report typical set size / cost.
            order = budget_seed_order(n_nodes, "random", config.rng(stream))
            seeds = select_seeds_under_budget(cost, budget, order=order)
            stream += 1
        elif strat == "greedy":
            seeds, _trace = greedy_influence_max(
                config, model, graph, cost, budget=budget,
                stream_offset=_GREEDY_STREAM_BASE,
            )
            reach = reach_fixed_seeds(
                config, model, seeds, n_runs=n_runs, stream_offset=stream
            )
            stream += n_runs
        else:  # hub / pagerank / betweenness
            scores = centrality_scores(
                graph, strat, config.rng(), k=config.betweenness_k
            )
            order = budget_seed_order(n_nodes, strat, config.rng(), scores=scores)
            seeds = select_seeds_under_budget(cost, budget, order=order)
            reach = reach_fixed_seeds(
                config, model, seeds, n_runs=n_runs, stream_offset=stream
            )
            stream += n_runs

        spent = float(cost[np.fromiter(seeds, dtype=np.int64)].sum()) if seeds else 0.0
        results.append(BudgetStrategyResult(
            strategy=strat, reach=reach,
            seed_set_size=len(seeds), spent_cost=spent,
        ))

    return BudgetExperimentResult(
        model_label=model.label, budget=float(budget),
        cost_alpha=float(cost_alpha), results=results,
    )
