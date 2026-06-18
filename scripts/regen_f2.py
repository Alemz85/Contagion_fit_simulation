"""Regenerate F2 and F3 with the fair (extended-p) threshold fit.

After the threshold p-grid robustness run, the threshold model's best (phi, p)
changed (p is no longer pinned at the 0.5 boundary). This script redraws both
the best-fit comparison (F2) and the KS-distance verdict heatmap (F3) for both
substrates, so the figures reflect the fair threshold best-fit:

    IC best    <- results/grid_robustness.json        (converged IC p)
    threshold  <- results/threshold_p_robustness.json (extended-p best phi,p)

Overwrites, for BA (results/) and WS (results_ws/):
    F2_best_fit_comparison.{png,svg}
    F3_distance_heatmap.{png,svg}

Run:  python scripts/regen_f2.py --n-nodes 50000 --n-runs 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contagion_fit import viz  # noqa: E402
from contagion_fit.config import Config  # noqa: E402
from contagion_fit.data import load_cascades  # noqa: E402
from contagion_fit.fit import (  # noqa: E402
    ComparisonResult, FitResult, decide_winner, ks_distance,
)
from contagion_fit.models import ComplexContagion, SimpleContagion  # noqa: E402
from contagion_fit.simulate import simulate_parallel  # noqa: E402

DATASETS = [
    ("higgs", "higgs (science news)"),
    ("twitter15", "twitter15 (rumors)"),
    ("twitter16", "twitter16 (rumors)"),
    ("weibo", "weibo (rumors)"),
]
SUBSTRATES = {"ba": "results", "ws": "results_ws"}


def _lookup(rows, substrate, dataset, key):
    for r in rows:
        if r["substrate"] == substrate and r["dataset"] == dataset:
            return r[key]
    raise KeyError(f"{substrate}/{dataset}/{key}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="regenerate F2 with fair threshold fit")
    p.add_argument("--n-nodes", type=int, default=50000)
    p.add_argument("--n-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ic_rows = json.loads((ROOT / "results" / "grid_robustness.json").read_text("utf-8"))["rows"]
    thr_rows = json.loads(
        (ROOT / "results" / "threshold_p_robustness.json").read_text("utf-8"))["rows"]

    for substrate, out_dir in SUBSTRATES.items():
        config = Config(substrate=substrate, n_nodes=args.n_nodes,
                        n_runs=args.n_runs, seed=args.seed, n_workers=args.workers,
                        results_dir=ROOT / out_dir)
        panels = []
        comparisons = []
        for offset, (dataset, title) in enumerate(DATASETS):
            observed = load_cascades(dataset)
            # Best parameters come from the (converged IC) and (fair threshold)
            # robustness runs; distances are recomputed here from the figure's
            # own n_runs simulation so F2 curves and F3 numbers are consistent.
            ic_p = _lookup(ic_rows, substrate, dataset, "ic_best_p")
            thr_params = _lookup(thr_rows, substrate, dataset, "threshold_best_params")
            ic_model = SimpleContagion(p=ic_p)
            thr_model = ComplexContagion(**thr_params)
            ic_sim = simulate_parallel(config, ic_model,
                                       stream_offset=50_000_000 + offset * args.n_runs)
            thr_sim = simulate_parallel(config, thr_model,
                                        stream_offset=60_000_000 + offset * args.n_runs)
            ic_d = ks_distance(observed, ic_sim)
            thr_d = ks_distance(observed, thr_sim)
            panels.append({
                "title": title, "observed": observed,
                "ic_sim": ic_sim, "thr_sim": thr_sim,
                "ic_label": ic_model.label, "thr_label": thr_model.label,
            })
            # Build a ComparisonResult so F3 reflects the new fair distances.
            winner, margin = decide_winner(ic_d, thr_d)
            comparisons.append(ComparisonResult(
                platform=dataset, label=None,
                ic=FitResult("SimpleContagion", {"p": ic_p}, ic_d, []),
                threshold=FitResult("ComplexContagion", thr_params, thr_d, []),
                winner=winner, margin=margin))
            print(f"{substrate} {dataset:10} IC {ic_model.label} d={ic_d:.3f}  |  "
                  f"Thr {thr_model.label} d={thr_d:.3f}  -> {winner}")
        f2 = viz.fig_best_fit_comparison(panels, config)
        f3 = viz.fig_distance_heatmap(comparisons, config)
        print(f"  -> wrote F2 {[p.name for p in f2]} and F3 {[p.name for p in f3]}")


if __name__ == "__main__":
    main()
