"""Grid-convergence / robustness check for the IC model.

In the headline run the Independent Cascade always selected the largest grid
value (p = 0.2), so its optimum was pinned at the boundary and the model
comparison could be unfair to IC. The threshold model, by contrast, already
optimised at an interior grid point and is treated as converged.

This script re-runs the IC grid search with an *extended* probability grid on
both substrates (BA and WS), for all four datasets, and checks:

    1. whether the IC optimum now falls at an interior grid point (converged)
       rather than at the boundary, and
    2. whether bracketing IC's true optimum changes any model-selection verdict
       relative to the converged threshold fit (re-used from the headline
       summary.json so the threshold side is not recomputed).

Run::

    python scripts/grid_robustness.py --n-nodes 50000 --n-runs 1000

Writes results/grid_robustness.json and prints a per-dataset table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the src layout importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contagion_fit.config import Config  # noqa: E402
from contagion_fit.data import load_cascades  # noqa: E402
from contagion_fit.fit import decide_winner, grid_search  # noqa: E402
from contagion_fit.models import SimpleContagion  # noqa: E402

# Extended IC grid: keeps the original points and adds 0.3, 0.5, 0.7 above the
# old 0.2 boundary so the optimum is bracketed from both sides.
EXTENDED_IC_P_GRID = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7)

DATASETS = ["higgs", "twitter15", "twitter16", "weibo"]
SUBSTRATES = {"ba": "results", "ws": "results_ws"}


def _threshold_from_summary(summary_dir: Path) -> dict[str, dict]:
    """Pull the converged threshold fit per platform from a headline summary."""
    data = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    out = {}
    for c in data["comparisons"]:
        out[c["platform"]] = {
            "best_params": c["threshold"]["best_params"],
            "best_distance": c["threshold"]["best_distance"],
        }
    return out


def _is_interior(best_p: float, grid: tuple[float, ...]) -> bool:
    return grid[0] < best_p < grid[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC grid robustness check")
    p.add_argument("--n-nodes", type=int, default=50000)
    p.add_argument("--n-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    print(f"{'substrate':<9} {'dataset':<11} {'IC p*':>6} {'interior':>9} "
          f"{'IC d':>7} {'Thr d':>7} {'winner':>9} {'changed':>8}")
    for substrate, summary_dir_name in SUBSTRATES.items():
        config = Config(
            substrate=substrate,
            n_nodes=args.n_nodes,
            n_runs=args.n_runs,
            seed=args.seed,
            n_workers=args.workers,
            ic_p_grid=EXTENDED_IC_P_GRID,
        )
        threshold = _threshold_from_summary(ROOT / summary_dir_name)
        original = json.loads(
            (ROOT / summary_dir_name / "summary.json").read_text(encoding="utf-8")
        )
        original_winner = {c["platform"]: c["winner"] for c in original["comparisons"]}

        for dataset in DATASETS:
            observed = load_cascades(dataset)
            ic = grid_search(
                config, SimpleContagion, {"p": EXTENDED_IC_P_GRID}, observed
            )
            thr = threshold[dataset]
            winner, margin = decide_winner(ic.best_distance, thr["best_distance"])
            interior = _is_interior(ic.best_params["p"], EXTENDED_IC_P_GRID)
            changed = winner != original_winner[dataset]
            rows.append({
                "substrate": substrate,
                "dataset": dataset,
                "ic_best_p": ic.best_params["p"],
                "ic_optimum_interior": interior,
                "ic_best_distance": ic.best_distance,
                "threshold_best_params": thr["best_params"],
                "threshold_best_distance": thr["best_distance"],
                "winner": winner,
                "margin": margin,
                "original_winner": original_winner[dataset],
                "winner_changed": changed,
                "ic_table": [
                    {"p": c.params["p"], "distance": c.distance} for c in ic.table
                ],
            })
            print(f"{substrate:<9} {dataset:<11} {ic.best_params['p']:>6g} "
                  f"{str(interior):>9} {ic.best_distance:>7.3f} "
                  f"{thr['best_distance']:>7.3f} {winner:>9} {str(changed):>8}")

    out_path = ROOT / "results" / "grid_robustness.json"
    out_path.write_text(
        json.dumps({"extended_ic_p_grid": list(EXTENDED_IC_P_GRID), "rows": rows},
                   indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")

    n_changed = sum(r["winner_changed"] for r in rows)
    n_interior = sum(r["ic_optimum_interior"] for r in rows)
    print(f"summary: {n_interior}/{len(rows)} IC optima now interior; "
          f"{n_changed}/{len(rows)} verdicts changed vs headline run")


if __name__ == "__main__":
    main()
