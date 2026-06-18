"""Robustness check for the threshold model's adoption probability p.

A code review pointed out that threshold_p_grid = (0.5, 1.0) is too coarse: the
threshold model's best p was pinned at the boundary p=0.5 in every WS cell (and
in BA/higgs). That is the same boundary-pinning we fixed for the IC p-grid but
never fixed for the threshold's adoption probability. Because lower p shrinks
cascades, it could cure the threshold model's WS overshoot and flip the verdict.

This script re-fits the threshold model with an extended p-grid (adding
0.1-0.4) on both substrates for all four datasets, and compares the new
threshold best-distance to the converged IC best-distance (re-used from
results/grid_robustness.json) to see whether any verdict changes.

Run:  python scripts/threshold_p_robustness.py --n-nodes 50000 --n-runs 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contagion_fit.config import Config  # noqa: E402
from contagion_fit.data import load_cascades  # noqa: E402
from contagion_fit.fit import decide_winner, grid_search  # noqa: E402
from contagion_fit.models import ComplexContagion  # noqa: E402

# Extended adoption-probability grid: keep the old {0.5, 1.0} and add lower
# values so the optimum can be bracketed from below.
EXTENDED_THRESHOLD_P = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)

DATASETS = {"higgs": "higgs", "twitter15": "twitter15",
            "twitter16": "twitter16", "weibo": "weibo"}
SUBSTRATES = {"ba": "results", "ws": "results_ws"}


def _ic_best(grid_robustness: dict, substrate: str, dataset: str) -> float:
    for r in grid_robustness["rows"]:
        if r["substrate"] == substrate and r["dataset"] == dataset:
            return r["ic_best_distance"]
    raise KeyError(f"no IC entry for {substrate}/{dataset}")


def _original_winner(summary_dir: Path, dataset: str) -> str:
    data = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    for c in data["comparisons"]:
        if c["platform"] == dataset:
            return c["winner"]
    return "?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="threshold p-grid robustness")
    p.add_argument("--n-nodes", type=int, default=50000)
    p.add_argument("--n-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--substrates", default="ba,ws",
                   help="comma list of substrates to run, e.g. 'ws'")
    p.add_argument("--phi-grid", default=None,
                   help="comma list overriding the phi grid, e.g. '0.1,0.15,0.2,0.3,0.4'")
    p.add_argument("--p-grid", default=None,
                   help="comma list overriding the threshold p grid, e.g. '0.1,0.2,0.3,0.5'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grid_rob = json.loads((ROOT / "results" / "grid_robustness.json").read_text(encoding="utf-8"))

    wanted = [s.strip() for s in args.substrates.split(",") if s.strip()]
    phi_override = (
        tuple(float(x) for x in args.phi_grid.split(",")) if args.phi_grid else None
    )
    p_grid = (
        tuple(float(x) for x in args.p_grid.split(",")) if args.p_grid
        else EXTENDED_THRESHOLD_P
    )

    rows = []
    print(f"{'sub':3} {'dataset':10} {'thr p*':>6} {'interior':>9} "
          f"{'thrD_new':>9} {'thrD_old':>9} {'ic D':>7} "
          f"{'winner_new':>11} {'was':>8} {'changed':>8}", flush=True)
    for substrate in wanted:
        summary_dir_name = SUBSTRATES[substrate]
        cfg_kw = dict(substrate=substrate, n_nodes=args.n_nodes, n_runs=args.n_runs,
                      seed=args.seed, n_workers=args.workers,
                      threshold_p_grid=p_grid)
        if phi_override is not None:
            cfg_kw["threshold_phi_grid"] = phi_override
        config = Config(**cfg_kw)
        summary_dir = ROOT / summary_dir_name
        old_summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
        old_thr = {c["platform"]: c["threshold"]["best_distance"]
                   for c in old_summary["comparisons"]}
        for dataset in DATASETS:
            observed = load_cascades(dataset)
            thr = grid_search(
                config, ComplexContagion,
                {"phi": config.threshold_phi_grid, "p": p_grid},
                observed,
            )
            ic_d = _ic_best(grid_rob, substrate, dataset)
            winner, margin = decide_winner(ic_d, thr.best_distance)
            was = _original_winner(summary_dir, dataset)
            interior = p_grid[0] < thr.best_params["p"] < p_grid[-1]
            changed = winner != was
            rows.append({
                "substrate": substrate, "dataset": dataset,
                "threshold_best_params": thr.best_params,
                "threshold_p_interior": interior,
                "threshold_best_distance_new": thr.best_distance,
                "threshold_best_distance_old": old_thr.get(dataset),
                "ic_best_distance": ic_d,
                "winner_new": winner, "winner_old": was,
                "winner_changed": changed, "margin": margin,
                "threshold_table": [
                    {"params": c.params, "distance": c.distance} for c in thr.table
                ],
            })
            print(f"{substrate:3} {dataset:10} {thr.best_params['p']:>6g} "
                  f"{str(interior):>9} {thr.best_distance:>9.3f} "
                  f"{old_thr.get(dataset, float('nan')):>9.3f} {ic_d:>7.3f} "
                  f"{winner:>11} {was:>8} {str(changed):>8}", flush=True)

    out = ROOT / "results" / "threshold_p_robustness.json"
    out.write_text(json.dumps({"extended_threshold_p_grid": list(EXTENDED_THRESHOLD_P),
                               "rows": rows}, indent=2), encoding="utf-8")
    n_changed = sum(r["winner_changed"] for r in rows)
    n_interior = sum(r["threshold_p_interior"] for r in rows)
    print(f"\nwrote {out}")
    print(f"summary: {n_interior}/{len(rows)} threshold-p optima interior; "
          f"{n_changed}/{len(rows)} verdicts changed vs headline")


if __name__ == "__main__":
    main()
