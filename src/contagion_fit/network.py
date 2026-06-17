"""Substrate networks: the stage on which contagion runs.

Three interchangeable substrates are provided and selected via ``Config``:

    "ba"    Barabasi-Albert scale-free graph (default)
    "ws"    Watts-Strogatz small-world graph (high clustering; favours
            complex contagion)
    "higgs" a sample of the real Higgs follower network, only available when
            ``higgs-social_network.edgelist.gz`` has been downloaded locally
            (see DATA_MANIFEST.md). Falls back with a clear error otherwise.

Running the same analysis on "ba" and "ws" is itself a sensitivity check on how
much the conclusions depend on the assumed substrate.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import networkx as nx
import numpy as np

from contagion_fit.config import Config, DATA_DIR

HIGGS_SOCIAL_EDGELIST = DATA_DIR / "higgs" / "higgs-social_network.edgelist.gz"


def build_network(config: Config, rng: np.random.Generator | None = None) -> nx.Graph:
    """Build the substrate graph selected in ``config``.

    The graph is returned with integer node labels ``0..n-1`` so that the
    simulation can use plain arrays/sets of ints.
    """
    if rng is None:
        rng = config.rng()

    kind = config.substrate.lower()
    if kind == "ba":
        return _barabasi_albert(config, rng)
    if kind == "ws":
        return _watts_strogatz(config, rng)
    if kind == "higgs":
        return _higgs_sample(config, rng)
    raise ValueError(f"unknown substrate {config.substrate!r}; expected ba|ws|higgs")


def _seed_int(rng: np.random.Generator) -> int:
    """networkx generators want an int seed; derive one from the injected rng."""
    return int(rng.integers(0, 2**31))


def _barabasi_albert(config: Config, rng: np.random.Generator) -> nx.Graph:
    return nx.barabasi_albert_graph(config.n_nodes, config.ba_m, seed=_seed_int(rng))


def _watts_strogatz(config: Config, rng: np.random.Generator) -> nx.Graph:
    return nx.watts_strogatz_graph(
        config.n_nodes, config.ws_k, config.ws_p, seed=_seed_int(rng)
    )


def _higgs_sample(config: Config, rng: np.random.Generator) -> nx.Graph:
    """Load the real Higgs follower network and take a connected sample.

    The full network (~456k nodes / 14.8M edges) is not shipped. If the file is
    missing we raise with an actionable message rather than silently degrading.
    """
    if not HIGGS_SOCIAL_EDGELIST.exists():
        raise FileNotFoundError(
            f"substrate 'higgs' needs {HIGGS_SOCIAL_EDGELIST.name}, which is not "
            "shipped. Download higgs-social_network.edgelist.gz from "
            "https://snap.stanford.edu/data/higgs-twitter.html into data/higgs/ "
            "(see DATA_MANIFEST.md)."
        )
    full = nx.Graph()
    with gzip.open(HIGGS_SOCIAL_EDGELIST, "rt") as f:
        for line in f:
            a, b = line.split()[:2]
            full.add_edge(int(a), int(b))
    return _snowball_sample(full, config.n_nodes, rng)


def _snowball_sample(graph: nx.Graph, target: int, rng: np.random.Generator) -> nx.Graph:
    """BFS snowball sample of ``target`` nodes, relabelled to ``0..k-1``.

    Snowball sampling keeps the local clustering and degree correlations of the
    source graph far better than node sampling does.
    """
    if graph.number_of_nodes() <= target:
        sample_nodes = list(graph.nodes())
    else:
        start = int(rng.choice(list(graph.nodes())))
        visited: set[int] = {start}
        frontier = [start]
        while frontier and len(visited) < target:
            node = frontier.pop(0)
            neighbours = list(graph.neighbors(node))
            rng.shuffle(neighbours)
            for nb in neighbours:
                if nb not in visited:
                    visited.add(nb)
                    frontier.append(nb)
                    if len(visited) >= target:
                        break
        sample_nodes = list(visited)
    sub = graph.subgraph(sample_nodes).copy()
    return nx.convert_node_labels_to_integers(sub)


def degree_ranked_nodes(graph: nx.Graph) -> np.ndarray:
    """Node ids sorted by degree, highest first (used for hub seeding)."""
    deg = np.asarray([d for _, d in graph.degree()])
    nodes = np.asarray([n for n, _ in graph.degree()])
    order = np.argsort(deg)[::-1]
    return nodes[order]


# --------------------------------------------------------------------------
# Seed-budget extension (Phase 4): a cost model and centrality scores.
#
# These power the budgeted seeding experiment that asks the Watts-Dodds
# question fairly: for a *fixed budget*, do many cheap ordinary accounts out-
# reach one expensive hub? They are pure functions of the graph, so they live
# here alongside degree_ranked_nodes.
# --------------------------------------------------------------------------

def degree_array(graph: nx.Graph) -> np.ndarray:
    """Degrees indexed by node id ``0..n-1`` (nodes are integer-labelled)."""
    n = graph.number_of_nodes()
    deg = np.zeros(n, dtype=np.int64)
    for v, d in graph.degree():
        deg[v] = d
    return deg


def degree_cost(graph: nx.Graph, alpha: float = 1.0) -> np.ndarray:
    """Per-node acquisition cost, rising with degree.

    ``cost(v) = 1 + alpha * degree(v) / mean_degree``. One unit is the baseline
    price of an *average*-degree account; a hub with ten times the mean degree
    costs about ``1 + 10*alpha`` units. ``alpha = 0`` makes every node cost 1,
    which reduces a fixed budget to a plain "k seeds" cap (so the same machinery
    covers both the multiple-seed and the cost-normalised comparisons).
    """
    deg = degree_array(graph).astype(np.float64)
    mean_deg = float(deg.mean()) or 1.0
    return 1.0 + alpha * deg / mean_deg


def centrality_scores(
    graph: nx.Graph,
    kind: str,
    rng: np.random.Generator | None = None,
    *,
    k: int | None = None,
) -> np.ndarray:
    """Centrality score per node id, used to rank candidate seeds.

    ``kind`` is one of ``"degree"`` / ``"hub"`` (degree), ``"pagerank"`` or
    ``"betweenness"``. Betweenness on large graphs uses ``k`` pivot samples
    (seeded from ``rng``) as an approximation; ``k = None`` is exact.
    """
    n = graph.number_of_nodes()
    kind = kind.lower()
    if kind in ("degree", "hub"):
        return degree_array(graph).astype(np.float64)
    if kind == "pagerank":
        pr = nx.pagerank(graph)
        scores = np.zeros(n, dtype=np.float64)
        for v, s in pr.items():
            scores[v] = s
        return scores
    if kind == "betweenness":
        seed = None if rng is None else int(rng.integers(0, 2**31))
        bc = nx.betweenness_centrality(graph, k=k, seed=seed)
        scores = np.zeros(n, dtype=np.float64)
        for v, s in bc.items():
            scores[v] = s
        return scores
    raise ValueError(
        f"unknown centrality {kind!r}; expected degree|hub|pagerank|betweenness"
    )
