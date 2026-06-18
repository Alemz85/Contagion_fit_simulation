"""Fitting and model selection: the heart of the project.

We never claim a model is "true". We ask which mechanism, at its best-fitting
parameters, reproduces the *shape* of the observed cascade-size distribution
more closely. The yardstick is the two-sample Kolmogorov-Smirnov statistic
computed on ``log10(size)`` so that differences in the heavy tail dominate (a KS
on raw sizes would be swamped by the bulk of tiny cascades).

Pipeline per dataset:
    1. grid_search the IC model over its probability grid -> best IC fit
    2. grid_search the threshold model over (phi, p) -> best threshold fit
    3. compare the two best KS distances -> declare a winner (or "tie")
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import ks_2samp

from contagion_fit.config import Config
from contagion_fit.models import ComplexContagion, SimpleContagion
from contagion_fit.simulate import simulate_parallel, simulation_pool

# Distances below this are treated as indistinguishable -> reported as a tie.
TIE_MARGIN = 0.02


def ks_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Two-sample KS statistic on ``log10(size)``.

    Sizes of zero are not expected (cascades have size >= 1). Empty inputs
    return ``inf`` so they never win a comparison.
    """
    if observed.size == 0 or simulated.size == 0:
        return float("inf")
    obs = np.log10(np.asarray(observed, dtype=np.float64))
    sim = np.log10(np.asarray(simulated, dtype=np.float64))
    return float(ks_2samp(obs, sim).statistic)


@dataclass(frozen=True)
class GridCell:
    """One evaluated point of the parameter grid."""

    params: dict[str, float]
    distance: float


@dataclass(frozen=True)
class FitResult:
    """Outcome of a grid search for a single model family."""

    model_name: str
    best_params: dict[str, float]
    best_distance: float
    table: list[GridCell] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_distance": self.best_distance,
            "table": [{"params": c.params, "distance": c.distance} for c in self.table],
        }


def _param_combinations(param_grid: dict[str, Sequence[float]]) -> list[dict[str, float]]:
    keys = list(param_grid)
    return [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]


def grid_search(
    config: Config,
    model_cls,
    param_grid: dict[str, Sequence[float]],
    observed: np.ndarray,
    n_runs: int | None = None,
    *,
    seed_strategy: str = "random",
) -> FitResult:
    """Search ``param_grid`` for the parameters minimising KS distance.

    Each grid cell gets its own random substream (via ``stream_offset``) so the
    search is reproducible and cells are mutually independent.
    """
    n_runs = config.n_runs if n_runs is None else n_runs
    combos = _param_combinations(param_grid)
    table: list[GridCell] = []
    best: GridCell | None = None
    # Open one worker pool for the whole search: the substrate graph is built
    # once per worker and reused across every cell (instead of respawning the
    # pool and rebuilding the graph for each parameter combination).
    with simulation_pool(config) as executor:
        for idx, params in enumerate(combos):
            model = model_cls(**params)
            simulated = simulate_parallel(
                config,
                model,
                n_runs=n_runs,
                seed_strategy=seed_strategy,
                stream_offset=idx * n_runs,
                executor=executor,
            )
            dist = ks_distance(observed, simulated)
            cell = GridCell(params=params, distance=dist)
            table.append(cell)
            if best is None or dist < best.distance:
                best = cell
    assert best is not None  # param_grid is never empty in practice
    return FitResult(
        model_name=model_cls.__name__,
        best_params=best.params,
        best_distance=best.distance,
        table=table,
    )


@dataclass(frozen=True)
class ComparisonResult:
    """IC best-fit vs threshold best-fit for one dataset/label."""

    platform: str
    label: str | None
    ic: FitResult
    threshold: FitResult
    winner: str  # "simple" | "complex" | "tie"
    margin: float

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "label": self.label,
            "winner": self.winner,
            "margin": self.margin,
            "ic": self.ic.to_dict(),
            "threshold": self.threshold.to_dict(),
        }


def decide_winner(ic_distance: float, threshold_distance: float) -> tuple[str, float]:
    """Return (winner, margin). ``margin`` is the absolute KS-distance gap."""
    margin = abs(ic_distance - threshold_distance)
    if margin < TIE_MARGIN:
        return "tie", margin
    return ("simple" if ic_distance < threshold_distance else "complex"), margin


def compare_models(
    config: Config,
    observed: np.ndarray,
    *,
    platform: str = "",
    label: str | None = None,
    n_runs: int | None = None,
) -> ComparisonResult:
    """Fit both model families to ``observed`` and declare a winner."""
    ic = grid_search(
        config,
        SimpleContagion,
        {"p": config.ic_p_grid},
        observed,
        n_runs=n_runs,
    )
    threshold = grid_search(
        config,
        ComplexContagion,
        {"phi": config.threshold_phi_grid, "p": config.threshold_p_grid},
        observed,
        n_runs=n_runs,
    )
    winner, margin = decide_winner(ic.best_distance, threshold.best_distance)
    return ComparisonResult(
        platform=platform,
        label=label,
        ic=ic,
        threshold=threshold,
        winner=winner,
        margin=margin,
    )
