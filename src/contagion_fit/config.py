"""Central parameter container.

Everything tunable lives here so that experiments are reproducible from a single
object. Random numbers are *never* drawn from a global state: callers build a
``numpy.random.Generator`` from ``Config.seed`` and inject it downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Repository layout: this file is src/contagion_fit/config.py, so the project
# root is three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

CASCADES_CSV = PROCESSED_DIR / "cascades_unified.csv"


@dataclass(frozen=True)
class Config:
    """Immutable run configuration.

    Use :meth:`rng` to obtain the project's only source of randomness. Because
    the dataclass is frozen, two runs built from the same ``Config`` are
    bit-for-bit reproducible.
    """

    # --- reproducibility ---
    seed: int = 20260612

    # --- substrate network (see network.py) ---
    substrate: str = "ba"  # "ba" | "ws" | "higgs"
    n_nodes: int = 50_000
    ba_m: int = 3
    ws_k: int = 6
    ws_p: float = 0.1

    # --- Monte-Carlo ---
    n_runs: int = 1000
    seed_count: int = 1
    n_workers: int | None = None  # None -> os.cpu_count()

    # --- grid search (fit.py) ---
    # The upper points (0.3, 0.5, 0.7) bracket the IC optimum from above: a
    # grid-convergence check (scripts/grid_robustness.py) showed the optimum is
    # interior, so the narrower (..., 0.2) grid used in the first run had merely
    # pinned IC at its boundary. See results/grid_robustness.json.
    ic_p_grid: tuple[float, ...] = (
        0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7
    )
    threshold_phi_grid: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
    threshold_p_grid: tuple[float, ...] = (0.5, 1.0)

    # --- flip-boundary sweep (experiments.py) ---
    flip_p_range: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1, 0.2)
    flip_phi_range: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4)
    flip_n_runs: int = 300

    # --- seed-budget experiment (Phase 4: experiments.seed_budget_experiment) ---
    # A fixed total budget is spent on seeds whose price rises with degree
    # (cost_alpha). Strategies compete on reach-per-cost, so "many cheap accounts
    # vs one expensive hub" is a fair fight (the Watts-Dodds test left open by
    # the single-seed seeding experiment).
    seed_budget: float = 12.0
    cost_alpha: float = 1.0
    seed_strategies: tuple[str, ...] = (
        "random", "hub", "pagerank", "betweenness", "greedy"
    )
    betweenness_k: int | None = 200  # pivot samples for approx betweenness
    greedy_eval_runs: int = 80       # MC runs per candidate during greedy search
    greedy_pool: int = 30            # candidate pool size (top-degree nodes)

    # --- output ---
    results_dir: Path = field(default=RESULTS_DIR)

    def rng(self, stream: int = 0) -> np.random.Generator:
        """Return an independent generator.

        ``stream`` lets independent components (e.g. parallel workers) draw from
        statistically independent substreams of the same root seed.
        """
        return np.random.default_rng([self.seed, stream])

    def child_seed(self, stream: int) -> int:
        """A deterministic integer seed for a worker/substream.

        Useful where a plain ``int`` seed is required (e.g. passed across a
        process boundary) rather than a ``Generator``.
        """
        ss = np.random.SeedSequence([self.seed, stream])
        return int(ss.generate_state(1, dtype=np.uint32)[0])
