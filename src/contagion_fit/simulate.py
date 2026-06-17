"""Monte-Carlo engine: turn a model + substrate into a size distribution.

One run = place ``seed_count`` seed nodes (uniformly at random, or on the
highest-degree hubs), run the model to termination, record the final cascade
size. Repeating this ``n_runs`` times yields the *simulated* cascade-size
distribution that is compared against the observed one in ``fit.py``.

Two entry points are provided:

``cascade_size_distribution``
    Serial, takes a prebuilt graph, matches the design signature exactly. Used
    by tests and wherever a graph is already in hand.

``simulate_parallel``
    Rebuilds the substrate inside each worker process (so the large graph is
    never pickled per task) and fans the runs out with ``ProcessPoolExecutor``.
    Each run draws from an independent substream of the configured seed, so the
    whole distribution is reproducible regardless of worker count.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Sequence

import networkx as nx
import numpy as np

from contagion_fit.config import Config
from contagion_fit.models import ContagionModel, _as_csr
from contagion_fit.network import build_network, degree_cost, degree_ranked_nodes

SeedStrategy = Literal["random", "hub"]


def choose_seeds(
    G: nx.Graph,
    strategy: SeedStrategy,
    seed_count: int,
    rng: np.random.Generator,
    *,
    ranked: np.ndarray | None = None,
) -> set[int]:
    """Pick the initial active set.

    ``ranked`` (precomputed degree-ranked node ids) can be passed in to avoid
    recomputing the degree order on every run for the hub strategy.
    """
    if strategy == "random":
        nodes = rng.choice(G.number_of_nodes(), size=seed_count, replace=False)
        return set(int(x) for x in nodes)
    if strategy == "hub":
        if ranked is None:
            ranked = degree_ranked_nodes(G)
        return set(int(x) for x in ranked[:seed_count])
    raise ValueError(f"unknown seed strategy {strategy!r}; expected random|hub")


def select_seeds_under_budget(
    cost: np.ndarray,
    budget: float,
    *,
    order: np.ndarray,
) -> set[int]:
    """Fill a seed set by walking ``order`` while staying within ``budget``.

    ``order`` is a priority over node ids (best first); ``cost[v]`` is node
    ``v``'s price. Nodes are added greedily in priority order, skipping any that
    would overshoot the remaining budget, until none fits. With unit costs this
    selects exactly ``floor(budget)`` nodes; with degree-rising costs an
    expensive hub crowds out many cheap accounts (and vice versa) -- which is
    the whole point of the comparison.
    """
    chosen: list[int] = []
    spent = 0.0
    for v in order:
        v = int(v)
        c = float(cost[v])
        if spent + c <= budget + 1e-9:
            chosen.append(v)
            spent += c
    return set(chosen)


def budget_seed_order(
    n_nodes: int,
    strategy: str,
    rng: np.random.Generator,
    *,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Priority order over node ids for a budgeted seeding ``strategy``.

    ``random`` shuffles uniformly (so cheap ordinary accounts are reached);
    every other strategy ranks by its precomputed ``scores`` descending.
    """
    if strategy == "random":
        return rng.permutation(n_nodes)
    if scores is None:
        raise ValueError(f"strategy {strategy!r} needs precomputed scores")
    return np.argsort(scores, kind="stable")[::-1]


def cascade_size_distribution(
    G: nx.Graph,
    model: ContagionModel,
    n_runs: int = 1000,
    seed_strategy: SeedStrategy = "random",
    seed_count: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Serial Monte-Carlo: return an array of ``n_runs`` final cascade sizes."""
    if rng is None:
        rng = np.random.default_rng(0)
    _as_csr(G)  # warm the adjacency cache once
    ranked = degree_ranked_nodes(G) if seed_strategy == "hub" else None
    sizes = np.empty(n_runs, dtype=np.int64)
    for i in range(n_runs):
        seeds = choose_seeds(G, seed_strategy, seed_count, rng, ranked=ranked)
        sizes[i] = len(model.run(G, seeds, rng))
    return sizes


# --------------------------------------------------------------------------
# Parallel driver. Workers rebuild the substrate once (in an initializer) so the
# graph is never serialised per task.
# --------------------------------------------------------------------------

_WORKER: dict[str, object] = {}


def _init_worker(config: Config) -> None:
    _WORKER.clear()  # clear stale keys from any previous initialisation
    graph = build_network(config)
    _as_csr(graph)
    _WORKER["graph"] = graph
    _WORKER["ranked"] = degree_ranked_nodes(graph)


def _run_one(task: tuple[ContagionModel, SeedStrategy, int, int]) -> int:
    model, strategy, seed_count, seed_int = task
    graph: nx.Graph = _WORKER["graph"]  # type: ignore[assignment]
    ranked: np.ndarray = _WORKER["ranked"]  # type: ignore[assignment]
    rng = np.random.default_rng(seed_int)
    seeds = choose_seeds(graph, strategy, seed_count, rng, ranked=ranked)
    return len(model.run(graph, seeds, rng))


def simulate_parallel(
    config: Config,
    model: ContagionModel,
    n_runs: int | None = None,
    seed_strategy: SeedStrategy = "random",
    seed_count: int | None = None,
    *,
    stream_offset: int = 0,
) -> np.ndarray:
    """Parallel Monte-Carlo driven by ``config``.

    ``stream_offset`` shifts the per-run substreams so that several calls (e.g.
    grid-search cells) draw non-overlapping randomness while staying fully
    reproducible.
    """
    n_runs = config.n_runs if n_runs is None else n_runs
    seed_count = config.seed_count if seed_count is None else seed_count
    n_workers = config.n_workers or os.cpu_count() or 1

    seeds_per_run = [
        config.child_seed(stream_offset + i) for i in range(n_runs)
    ]
    tasks = [(model, seed_strategy, seed_count, s) for s in seeds_per_run]

    if n_workers == 1:
        _init_worker(config)
        return np.asarray([_run_one(t) for t in tasks], dtype=np.int64)

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(config,)
    ) as pool:
        sizes = list(pool.map(_run_one, tasks, chunksize=max(1, n_runs // (n_workers * 4))))
    return np.asarray(sizes, dtype=np.int64)


# --------------------------------------------------------------------------
# Budgeted-seeding Monte-Carlo drivers (Phase 4).
#
# Two cases:
#   * a *fixed* seed set (the deterministic strategies -- hub, pagerank,
#     betweenness, greedy -- pick one set up front, so only the cascade dynamics
#     are random across runs);
#   * *random* seeding under a budget, which redraws a fresh cheap-account set
#     every run.
# Both keep the per-run substream = child_seed(stream_offset + i), so the whole
# distribution stays reproducible regardless of worker count.
# --------------------------------------------------------------------------

def reach_fixed_seeds_on_graph(
    G: nx.Graph,
    model: ContagionModel,
    seeds: set[int],
    child_seeds: Sequence[int],
) -> np.ndarray:
    """Serial in-process reach for a fixed seed set (used by the greedy search).

    Takes a prebuilt graph so the (expensive) substrate is not rebuilt per
    marginal-gain evaluation.
    """
    _as_csr(G)
    return np.asarray(
        [len(model.run(G, seeds, np.random.default_rng(s))) for s in child_seeds],
        dtype=np.int64,
    )


def _init_budget_worker(config: Config, cost_alpha: float) -> None:
    _WORKER.clear()  # clear stale keys from any previous initialisation
    graph = build_network(config)
    _as_csr(graph)
    _WORKER["graph"] = graph
    _WORKER["cost"] = degree_cost(graph, cost_alpha)


def _run_fixed(task: tuple[ContagionModel, tuple[int, ...], int]) -> int:
    model, seeds_tuple, seed_int = task
    graph: nx.Graph = _WORKER["graph"]  # type: ignore[assignment]
    rng = np.random.default_rng(seed_int)
    return len(model.run(graph, set(seeds_tuple), rng))


def _run_random_budget(task: tuple[ContagionModel, float, int]) -> int:
    model, budget, seed_int = task
    graph: nx.Graph = _WORKER["graph"]  # type: ignore[assignment]
    cost: np.ndarray = _WORKER["cost"]  # type: ignore[assignment]
    rng = np.random.default_rng(seed_int)
    order = budget_seed_order(graph.number_of_nodes(), "random", rng)
    seeds = select_seeds_under_budget(cost, budget, order=order)
    return len(model.run(graph, seeds, rng))


def reach_fixed_seeds(
    config: Config,
    model: ContagionModel,
    seeds: set[int],
    *,
    n_runs: int | None = None,
    stream_offset: int = 0,
) -> np.ndarray:
    """Parallel reach distribution for a single, fixed seed set."""
    n_runs = config.n_runs if n_runs is None else n_runs
    n_workers = config.n_workers or os.cpu_count() or 1
    seeds_tuple = tuple(sorted(int(s) for s in seeds))
    tasks = [
        (model, seeds_tuple, config.child_seed(stream_offset + i))
        for i in range(n_runs)
    ]
    if n_workers == 1:
        _init_worker(config)
        return np.asarray([_run_fixed(t) for t in tasks], dtype=np.int64)
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(config,)
    ) as pool:
        sizes = list(pool.map(_run_fixed, tasks, chunksize=max(1, n_runs // (n_workers * 4))))
    return np.asarray(sizes, dtype=np.int64)


def reach_random_budget(
    config: Config,
    model: ContagionModel,
    budget: float,
    *,
    cost_alpha: float | None = None,
    n_runs: int | None = None,
    stream_offset: int = 0,
) -> np.ndarray:
    """Parallel reach distribution for random seeding under a fixed budget.

    A fresh random seed set (cheap ordinary accounts, capped by ``budget``) is
    drawn each run, so this captures both the seed-choice and the dynamic
    randomness.
    """
    cost_alpha = config.cost_alpha if cost_alpha is None else cost_alpha
    n_runs = config.n_runs if n_runs is None else n_runs
    n_workers = config.n_workers or os.cpu_count() or 1
    tasks = [
        (model, float(budget), config.child_seed(stream_offset + i))
        for i in range(n_runs)
    ]
    if n_workers == 1:
        _init_budget_worker(config, cost_alpha)
        return np.asarray([_run_random_budget(t) for t in tasks], dtype=np.int64)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_budget_worker,
        initargs=(config, cost_alpha),
    ) as pool:
        sizes = list(
            pool.map(_run_random_budget, tasks, chunksize=max(1, n_runs // (n_workers * 4)))
        )
    return np.asarray(sizes, dtype=np.int64)
