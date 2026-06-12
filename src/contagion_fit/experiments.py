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
from contagion_fit.simulate import simulate_parallel


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
