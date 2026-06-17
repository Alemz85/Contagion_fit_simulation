"""Figure generation (F1-F5).

Only matplotlib is used. Every figure is written to ``results/`` as both SVG
(for the slides) and PNG (for quick preview). The log-log CCDF is the recurring
visual idiom, and each figure is annotated with the substrate, the number of
Monte-Carlo runs and the parameters so a reader can reproduce it.

These functions only *plot*; they take already-computed arrays/results so that
the (slow) simulation and the (fast) drawing stay decoupled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contagion_fit.config import Config  # noqa: E402
from contagion_fit.data import ccdf  # noqa: E402
from contagion_fit.experiments import (  # noqa: E402
    BudgetExperimentResult,
    FlipBoundaryResult,
    SeedingResult,
)
from contagion_fit.fit import ComparisonResult  # noqa: E402


def _save(fig: plt.Figure, results_dir: Path, name: str) -> list[Path]:
    """Write ``fig`` as both SVG and PNG; return the paths written."""
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        path = results_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_ccdf(ax: plt.Axes, sizes: np.ndarray, **kwargs) -> None:
    x, p = ccdf(sizes)
    if x.size:
        ax.loglog(x, p, **kwargs)


# --------------------------------------------------------------------------
# F1 - observed CCDFs overlaid
# --------------------------------------------------------------------------
def fig_observed_ccdf(
    observed_by_dataset: Mapping[str, np.ndarray],
    config: Config,
    name: str = "F1_observed_ccdf",
) -> list[Path]:
    """Overlay the observed cascade-size CCDF of every dataset (log-log)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset, sizes in observed_by_dataset.items():
        _plot_ccdf(ax, sizes, marker=".", linestyle="-", markersize=4,
                   label=f"{dataset} (n={sizes.size:,})")
    ax.set_xlabel("cascade size x")
    ax.set_ylabel("P(X >= x)")
    ax.set_title("F1 - Observed cascade-size CCDF by dataset")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    return _save(fig, config.results_dir, name)


# --------------------------------------------------------------------------
# F2 - observed vs best-fit IC vs best-fit threshold, one panel per dataset
# --------------------------------------------------------------------------
def fig_best_fit_comparison(
    panels: Sequence[Mapping[str, object]],
    config: Config,
    name: str = "F2_best_fit_comparison",
) -> list[Path]:
    """2x2 grid; each panel overlays observed, IC-best and threshold-best CCDFs.

    Each entry of ``panels`` is a mapping with keys:
        title (str), observed (ndarray), ic_sim (ndarray), thr_sim (ndarray),
        ic_label (str), thr_label (str).
    """
    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows), squeeze=False)
    for idx, panel in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        _plot_ccdf(ax, panel["observed"], color="black", marker=".",
                   linestyle="none", markersize=4, label="observed")
        _plot_ccdf(ax, panel["ic_sim"], color="tab:blue", linestyle="-",
                   label=panel.get("ic_label", "IC best"))
        _plot_ccdf(ax, panel["thr_sim"], color="tab:red", linestyle="--",
                   label=panel.get("thr_label", "Threshold best"))
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel("cascade size x")
        ax.set_ylabel("P(X >= x)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    for idx in range(n, rows * cols):  # hide unused axes
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(
        f"F2 - Best-fit comparison ({config.substrate.upper()} substrate, "
        f"n_runs={config.n_runs})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, config.results_dir, name)


# --------------------------------------------------------------------------
# F3 - KS-distance heatmap (datasets x models)
# --------------------------------------------------------------------------
def fig_distance_heatmap(
    comparisons: Sequence[ComparisonResult],
    config: Config,
    name: str = "F3_distance_heatmap",
) -> list[Path]:
    """Matrix of best KS distances: rows = datasets, cols = {IC, Threshold}."""
    labels = [f"{c.platform}/{c.label}" if c.label else c.platform for c in comparisons]
    matrix = np.array(
        [[c.ic.best_distance, c.threshold.best_distance] for c in comparisons]
    )
    fig, ax = plt.subplots(figsize=(6, 0.55 * len(comparisons) + 2))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1], ["IC (simple)", "Threshold (complex)"])
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i, c in enumerate(comparisons):
        for j, val in enumerate(matrix[i]):
            winner_here = (j == 0 and c.winner == "simple") or (
                j == 1 and c.winner == "complex"
            )
            ax.text(j, i, f"{val:.3f}" + ("*" if winner_here else ""),
                    ha="center", va="center",
                    color="white" if val < matrix.max() * 0.6 else "black",
                    fontsize=8, fontweight="bold" if winner_here else "normal")
    ax.set_title("F3 - KS distance (lower = better fit; * = winner)")
    fig.colorbar(im, ax=ax, label="KS distance on log10(size)")
    fig.tight_layout()
    return _save(fig, config.results_dir, name)


# --------------------------------------------------------------------------
# F4 - seeding experiment: hub vs random reach with error bars
# --------------------------------------------------------------------------
def fig_seeding(
    results: Sequence[SeedingResult],
    config: Config,
    name: str = "F4_seeding",
) -> list[Path]:
    """Grouped bar chart of mean reach (+/- SEM) for hub vs random seeding."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(results))
    width = 0.38
    hub_means = [r.hub.mean() for r in results]
    hub_err = [r.hub.std() / np.sqrt(len(r.hub)) for r in results]
    rnd_means = [r.random.mean() for r in results]
    rnd_err = [r.random.std() / np.sqrt(len(r.random)) for r in results]
    ax.bar(x - width / 2, hub_means, width, yerr=hub_err, capsize=4,
           label="hub seed", color="tab:orange")
    ax.bar(x + width / 2, rnd_means, width, yerr=rnd_err, capsize=4,
           label="random seed", color="tab:gray")
    ax.set_xticks(x, [r.model_label for r in results], rotation=15, fontsize=8)
    ax.set_ylabel("expected reach (final cascade size)")
    ax.set_title(f"F4 - Seeding strategy ({config.substrate.upper()}, "
                 f"n_runs={config.n_runs})")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    return _save(fig, config.results_dir, name)


# --------------------------------------------------------------------------
# F5 - flip-boundary heatmap
# --------------------------------------------------------------------------
def fig_flip_boundary(
    result: FlipBoundaryResult,
    config: Config,
    name: str = "F5_flip_boundary",
) -> list[Path]:
    """Heatmap of hub advantage over the (p, phi) plane.

    A diverging colour map centred at zero makes the sign-change boundary - where
    the better seeding strategy flips - immediately visible.
    """
    adv = result.advantage
    vmax = float(np.abs(adv).max()) or 1.0
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(adv, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="lower")
    ax.set_xticks(range(len(result.p_range)), [f"{p:g}" for p in result.p_range])
    ax.set_yticks(range(len(result.phi_range)), [f"{phi:g}" for phi in result.phi_range])
    ax.set_xlabel("transmission probability p")
    ax.set_ylabel("threshold phi")
    for i in range(adv.shape[0]):
        for j in range(adv.shape[1]):
            ax.text(j, i, f"{adv[i, j]:.0f}", ha="center", va="center",
                    fontsize=7,
                    color="black" if abs(adv[i, j]) < vmax * 0.5 else "white")
    ax.set_title("F5 - Hub advantage = mean(hub) - mean(random)\n"
                 "(red: hubs win, blue: random wins)")
    fig.colorbar(im, ax=ax, label="hub advantage (nodes)")
    fig.tight_layout()
    return _save(fig, config.results_dir, name)


# --------------------------------------------------------------------------
# F8 - seed-budget experiment: reach and reach-per-cost by strategy
# --------------------------------------------------------------------------
def fig_budget_cost(
    result: BudgetExperimentResult,
    config: Config,
    name: str = "F8_budget_cost",
) -> list[Path]:
    """Two-panel comparison of seeding strategies at a fixed budget.

    Left: mean reach (+/- Monte-Carlo SEM), the headline "who reaches most for
    the same spend". Right: reach-per-cost, the efficiency view that exposes any
    regime where a distributed crowd of cheap accounts beats one expensive hub.
    The winning bar in each panel is highlighted.
    """
    res = result.results
    strategies = [r.strategy for r in res]
    x = np.arange(len(res))

    means = np.array([r.mean_reach for r in res])
    sems = np.array([r.sem for r in res])
    rpc = np.array([r.reach_per_cost for r in res])
    sizes = [r.seed_set_size for r in res]

    best_reach = int(np.argmax(means))
    best_rpc = int(np.argmax(rpc))
    base = "tab:gray"
    win = "tab:green"

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

    bars_l = axl.bar(x, means, yerr=sems, capsize=4,
                     color=[win if i == best_reach else base for i in range(len(res))])
    axl.set_xticks(x, strategies, rotation=20, fontsize=9)
    axl.set_ylabel("expected reach (final cascade size)")
    axl.set_title("Reach at fixed budget (+/- MC SEM)")
    axl.grid(True, axis="y", ls=":", alpha=0.4)
    for i, b in enumerate(bars_l):
        axl.annotate(f"k={sizes[i]}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8,
                     xytext=(0, 2), textcoords="offset points")

    bars_r = axr.bar(x, rpc, color=[win if i == best_rpc else base for i in range(len(res))])
    axr.set_xticks(x, strategies, rotation=20, fontsize=9)
    axr.set_ylabel("reach per unit cost")
    axr.set_title("Efficiency: reach / spend")
    axr.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        f"F8 - Seed-budget comparison ({config.substrate.upper()} substrate, "
        f"n={config.n_nodes:,}, {result.model_label}, budget={result.budget:g}, "
        f"cost_alpha={result.cost_alpha:g}, n_runs={config.n_runs})\n"
        f"reach winner: {strategies[best_reach]}  |  "
        f"reach-per-cost winner: {strategies[best_rpc]}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, config.results_dir, name)
