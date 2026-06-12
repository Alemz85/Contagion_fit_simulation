"""Contagion models with a shared interface.

Both models implement the same protocol::

    run(G, seeds, rng) -> set[int]   # the final set of activated nodes

and have no side effects: they never mutate ``G``, ``seeds`` or any global
state, and all randomness comes from the injected ``rng``.

SimpleContagion is the Independent Cascade (IC) model: each newly activated node
gets exactly one chance to activate each inactive neighbour, with probability
``p`` per edge. This is the canonical *simple* contagion.

ComplexContagion is a fractional-threshold model (Watts 2002 family): a node
activates once the fraction of its neighbours that are active reaches ``phi``;
on crossing the threshold it adopts with probability ``p`` (``p = 1`` is the
deterministic Watts threshold model). This captures *complex* contagion, where
multiple reinforcing contacts are required.

For speed on large graphs the neighbour lists are precomputed once into a CSR
(compressed sparse row) layout and cached on the graph object; both models read
from that cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np

_CSR_ATTR = "_contagion_fit_csr"


def _as_csr(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (indptr, indices, degree) CSR adjacency, cached on ``G``.

    ``indices[indptr[v]:indptr[v+1]]`` are the neighbours of node ``v``. Nodes
    are assumed to be integers ``0..n-1`` (guaranteed by network.build_network).
    """
    cached = G.graph.get(_CSR_ATTR)
    if cached is not None:
        return cached
    n = G.number_of_nodes()
    degree = np.zeros(n, dtype=np.int64)
    for v, d in G.degree():
        degree[v] = d
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    cursor = indptr[:-1].copy()
    for u, v in G.edges():
        indices[cursor[u]] = v
        cursor[u] += 1
        indices[cursor[v]] = u
        cursor[v] += 1
    csr = (indptr, indices, degree)
    G.graph[_CSR_ATTR] = csr
    return csr


@runtime_checkable
class ContagionModel(Protocol):
    """Common interface for all contagion mechanisms."""

    def run(self, G: nx.Graph, seeds: set[int], rng: np.random.Generator) -> set[int]:
        """Return the set of nodes active when the cascade has terminated."""
        ...

    @property
    def label(self) -> str:
        """Human-readable identifier including parameters (for figures)."""
        ...


@dataclass(frozen=True)
class SimpleContagion:
    """Independent Cascade. Each edge fires at most once with probability ``p``."""

    p: float

    @property
    def label(self) -> str:
        return f"IC(p={self.p:g})"

    def run(self, G: nx.Graph, seeds: set[int], rng: np.random.Generator) -> set[int]:
        indptr, indices, _ = _as_csr(G)
        n = indptr.size - 1
        active = np.zeros(n, dtype=bool)
        seed_arr = np.fromiter(seeds, dtype=np.int64)
        active[seed_arr] = True
        frontier = seed_arr
        p = self.p
        while frontier.size:
            # Gather all out-edges from the current frontier in one shot.
            starts = indptr[frontier]
            ends = indptr[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            # Flatten neighbour indices of the whole frontier.
            offsets = np.repeat(starts, counts) + (
                np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            )
            cand = indices[offsets]
            # One independent trial per edge.
            success = rng.random(total) < p
            cand = cand[success]
            if cand.size == 0:
                break
            newly = np.unique(cand[~active[cand]])
            if newly.size == 0:
                break
            active[newly] = True
            frontier = newly
        return set(np.flatnonzero(active).tolist())


@dataclass(frozen=True)
class ComplexContagion:
    """Fractional threshold model.

    A node activates once at least a fraction ``phi`` of its neighbours are
    active; adoption on crossing is stochastic with probability ``p``
    (``p = 1`` -> deterministic). Updates are synchronous and iterated until no
    further change.
    """

    phi: float
    p: float = 1.0

    @property
    def label(self) -> str:
        suffix = "" if self.p == 1.0 else f",p={self.p:g}"
        return f"Threshold(phi={self.phi:g}{suffix})"

    def run(self, G: nx.Graph, seeds: set[int], rng: np.random.Generator) -> set[int]:
        indptr, indices, degree = _as_csr(G)
        n = indptr.size - 1
        active = np.zeros(n, dtype=bool)
        active[np.fromiter(seeds, dtype=np.int64)] = True

        # Required active-neighbour count per node: ceil(phi * degree).
        with np.errstate(invalid="ignore"):
            need = np.ceil(self.phi * degree).astype(np.int64)
        # A node with phi > 0 but degree 0 can never reach threshold; with
        # phi == 0 every node (need == 0) activates immediately.
        need = np.maximum(need, 1) if self.phi > 0 else np.zeros(n, dtype=np.int64)

        while True:
            # Count active neighbours for every node via CSR scatter-add.
            active_idx = np.flatnonzero(active)
            if active_idx.size == n:
                break
            counts = _neighbour_active_counts(indptr, indices, active, n)
            eligible = (counts >= need) & ~active
            cand = np.flatnonzero(eligible)
            if cand.size == 0:
                break
            if self.p < 1.0:
                cand = cand[rng.random(cand.size) < self.p]
                if cand.size == 0:
                    break
            active[cand] = True
        return set(np.flatnonzero(active).tolist())


def _neighbour_active_counts(
    indptr: np.ndarray, indices: np.ndarray, active: np.ndarray, n: int
) -> np.ndarray:
    """For each node, how many of its neighbours are currently active."""
    active_nodes = np.flatnonzero(active)
    if active_nodes.size == 0:
        return np.zeros(n, dtype=np.int64)
    starts = indptr[active_nodes]
    ends = indptr[active_nodes + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(n, dtype=np.int64)
    offsets = np.repeat(starts, counts) + (
        np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    )
    # Each active node contributes +1 to each of its neighbours.
    contributions = indices[offsets]
    return np.bincount(contributions, minlength=n).astype(np.int64)
