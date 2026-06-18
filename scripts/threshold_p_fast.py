"""Fast threshold p-grid robustness: simulate each cell once, score all datasets.

Equivalent to threshold_p_robustness.py but ~4x faster: the threshold
simulation for a given (phi, p) cell depends only on the substrate graph, not on
which observed dataset we fit. So we simulate each of the 12 cells ONCE (reusing
one worker pool), cache the cascade sizes, then score every dataset against the
cached sizes. Stream offsets match grid_search (idx*n_runs), so the numbers are
bit-identical to the slower per-dataset version.

Results are merged into results/threshold_p_robustness.json (rows for the given
substrate replace any existing ones), so F2/F3 regeneration sees both substrates.

Run:  python scripts/threshold_p_fast.py --substrate ba --n-runs 500 --workers 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contagion_fit.config import Config  # noqa: E402
from contagion_fit.data import load_cascades  # noqa: E402
from contagion_fit.fit import decide_winner, ks_distance  # noqa: E402
from contagion_fit.models import ComplexContagion  # noqa: E402
from contagion_fit.simulate import simulate_parallel, simulation_pool  # noqa: E402

DEFAULT_PHI = (0.15, 0.2, 0.3)
# p must bracket BOTH optima: WS wants low p (~0.2), BA rumors want p=1.0.
DEFAULT_P = (0.1, 0.2, 0.3, 0.5, 1.0)
DATASETS = ["higgs", "twitter15", "twitter16", "weibo"]
SUBSTRATES = {"ba": "results", "ws": "results_ws"}
OUT = ROOT / "results" / "threshold_p_robustness.json"


def _ic_best(grid_rob: dict, substrate: str, dataset: str) -> float:
    for r in grid_rob["rows"]:
        if r["substrate"] == substrate and r["dataset"] == dataset:
            return r["ic_best_distance"]
    raise KeyError(f"{substrate}/{dataset}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fast threshold p robustness")
    p.add_argument("--substrate", default="ba")
    p.add_argument("--n-nodes", type=int, default=50000)
    p.add_argument("--n-runs", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--phi-grid", default=None, help="comma list overriding phi")
    p.add_argument("--p-grid", default=None, help="comma list overriding p")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    substrate = args.substrate
    n_runs = args.n_runs
    PHI = (tuple(float(x) for x in args.phi_grid.split(",")) if args.phi_grid
           else DEFAULT_PHI)
    P_GRID = (tuple(float(x) for x in args.p_grid.split(",")) if args.p_grid
              else DEFAULT_P)
    config = Config(substrate=substrate, n_nodes=args.n_nodes, n_runs=n_runs,
                    seed=args.seed, n_workers=args.workers,
                    threshold_phi_grid=PHI, threshold_p_grid=P_GRID)

    combos = list(itertools.product(PHI, P_GRID))  # phi-major, matches grid_search
    print(f"[{substrate}] simulating {len(combos)} cells once "
          f"(phi {PHI} x p {P_GRID}), n_runs={n_runs}", flush=True)

    t0 = time.time()
    cached = []  # (phi, p, sizes)
    with simulation_pool(config) as ex:
        for idx, (phi, p) in enumerate(combos):
            sizes = simulate_parallel(
                config, ComplexContagion(phi=phi, p=p), n_runs=n_runs,
                executor=ex, stream_offset=idx * n_runs,
            )
            cached.append((phi, p, sizes))
    print(f"[{substrate}] simulation done in {time.time() - t0:.0f} s", flush=True)

    grid_rob = json.loads((ROOT / "results" / "grid_robustness.json").read_text("utf-8"))
    summary_dir = ROOT / SUBSTRATES[substrate]
    old_summary = json.loads((summary_dir / "summary.json").read_text("utf-8"))
    old_thr = {c["platform"]: c["threshold"]["best_distance"]
               for c in old_summary["comparisons"]}
    old_winner = {c["platform"]: c["winner"] for c in old_summary["comparisons"]}

    rows = []
    print(f"{'sub':3} {'dataset':10} {'thr p*':>6} {'thrD_new':>9} {'thrD_old':>9} "
          f"{'ic D':>7} {'winner_new':>11} {'was':>8} {'changed':>8}", flush=True)
    for dataset in DATASETS:
        observed = load_cascades(dataset)
        scored = [(phi, p, ks_distance(observed, sizes)) for phi, p, sizes in cached]
        phi_b, p_b, thr_d = min(scored, key=lambda t: t[2])
        ic_d = _ic_best(grid_rob, substrate, dataset)
        winner, margin = decide_winner(ic_d, thr_d)
        was = old_winner.get(dataset, "?")
        interior = P_GRID[0] < p_b < P_GRID[-1]
        rows.append({
            "substrate": substrate, "dataset": dataset,
            "threshold_best_params": {"phi": phi_b, "p": p_b},
            "threshold_p_interior": interior,
            "threshold_best_distance_new": thr_d,
            "threshold_best_distance_old": old_thr.get(dataset),
            "ic_best_distance": ic_d,
            "winner_new": winner, "winner_old": was,
            "winner_changed": winner != was, "margin": margin,
            "threshold_table": [
                {"params": {"phi": phi, "p": p}, "distance": d} for phi, p, d in scored
            ],
        })
        print(f"{substrate:3} {dataset:10} {p_b:>6g} {thr_d:>9.3f} "
              f"{old_thr.get(dataset, float('nan')):>9.3f} {ic_d:>7.3f} "
              f"{winner:>11} {was:>8} {str(winner != was):>8}", flush=True)

    existing = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {"rows": []}
    kept = [r for r in existing.get("rows", []) if r["substrate"] != substrate]
    OUT.write_text(json.dumps(
        {"extended_threshold_p_grid": list(P_GRID), "phi_grid": list(PHI),
         "rows": kept + rows}, indent=2), encoding="utf-8")
    print(f"\nmerged {len(rows)} {substrate} rows into {OUT.name} "
          f"(kept {len(kept)} other-substrate rows)", flush=True)


if __name__ == "__main__":
    main()
