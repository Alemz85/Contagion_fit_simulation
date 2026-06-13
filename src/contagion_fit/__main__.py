"""End-to-end pipeline: ``python -m contagion_fit``.

Runs the full analysis described in DESIGN.md section 7:

    1. load observed cascade sizes for each dataset
    2. fit IC and threshold models to each (grid search + KS)
    3. run the seeding experiment and the flip-boundary sweep
    4. write figures F1-F5 and results/summary.json

Defaults are deliberately light (small substrate / few runs) so the command
finishes quickly for a smoke run; pass flags to scale up for the real figures.
Example for the full run::

    python -m contagion_fit --n-nodes 50000 --n-runs 1000 --substrate ba
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from contagion_fit import viz
from contagion_fit.config import Config
from contagion_fit.data import load_cascades, summary
from contagion_fit.experiments import (
    flip_boundary,
    seed_budget_experiment,
    seeding_experiment,
)
from contagion_fit.fit import compare_models
from contagion_fit.models import ComplexContagion, SimpleContagion
from contagion_fit.simulate import simulate_parallel

# The four datasets overlaid in F1/F2 (platform, observed-label-or-None, title).
DATASETS = [
    ("higgs", None, "higgs (science news)"),
    ("twitter15", None, "twitter15 (rumors)"),
    ("twitter16", None, "twitter16 (rumors)"),
    ("weibo", None, "weibo (rumors)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="contagion-fit full pipeline")
    p.add_argument("--substrate", default="ba", choices=["ba", "ws", "higgs"])
    p.add_argument("--n-nodes", type=int, default=2000)
    p.add_argument("--n-runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--results", type=Path, default=None)
    # Phase 4: budgeted seeding experiment (opt-in; adds F8). Off by default so
    # existing runs are unchanged and the (slower) greedy search is not forced.
    p.add_argument("--budget-experiment", action="store_true",
                   help="run the seed-budget / cost experiment and emit F8")
    p.add_argument("--seed-budget", type=float, default=None,
                   help="total seeding budget (default: Config.seed_budget)")
    p.add_argument("--cost-alpha", type=float, default=None,
                   help="how steeply seed cost rises with degree (0 = flat)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    kwargs = dict(
        substrate=args.substrate,
        n_nodes=args.n_nodes,
        n_runs=args.n_runs,
        seed=args.seed,
        n_workers=args.workers,
    )
    if args.results is not None:
        kwargs["results_dir"] = args.results
    if args.seed_budget is not None:
        kwargs["seed_budget"] = args.seed_budget
    if args.cost_alpha is not None:
        kwargs["cost_alpha"] = args.cost_alpha
    return Config(**kwargs)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] {config}")

    # --- observed data ---
    observed_by_dataset = {
        title: load_cascades(platform, label) for platform, label, title in DATASETS
    }
    print("[F1] observed CCDF")
    viz.fig_observed_ccdf(observed_by_dataset, config)

    # --- fit both models to each dataset ---
    comparisons = []
    panels = []
    for platform, label, title in DATASETS:
        observed = observed_by_dataset[title]
        print(f"[fit] {title}: {summary(observed)}")
        cmp = compare_models(config, observed, platform=platform, label=label)
        comparisons.append(cmp)
        print(f"      IC best {cmp.ic.best_params} d={cmp.ic.best_distance:.3f} | "
              f"Threshold best {cmp.threshold.best_params} "
              f"d={cmp.threshold.best_distance:.3f} -> {cmp.winner}")
        # Re-simulate at the best parameters to draw the fitted CCDFs (F2).
        ic_sim = simulate_parallel(
            config, SimpleContagion(**cmp.ic.best_params), stream_offset=10_000_000
        )
        thr_sim = simulate_parallel(
            config, ComplexContagion(**cmp.threshold.best_params),
            stream_offset=20_000_000,
        )
        panels.append({
            "title": title,
            "observed": observed,
            "ic_sim": ic_sim,
            "thr_sim": thr_sim,
            "ic_label": cmp.ic.best_params and SimpleContagion(**cmp.ic.best_params).label,
            "thr_label": ComplexContagion(**cmp.threshold.best_params).label,
        })

    print("[F2] best-fit comparison")
    viz.fig_best_fit_comparison(panels, config)
    print("[F3] distance heatmap")
    viz.fig_distance_heatmap(comparisons, config)

    # --- seeding experiment (F4): one IC and one threshold model ---
    print("[F4] seeding experiment")
    seeding_results = [
        seeding_experiment(config, SimpleContagion(p=0.05)),
        seeding_experiment(config, ComplexContagion(phi=0.2)),
    ]
    viz.fig_seeding(seeding_results, config)

    # --- flip boundary (F5) ---
    print("[F5] flip boundary")
    flip = flip_boundary(config)
    viz.fig_flip_boundary(flip, config)

    # --- seed-budget experiment (F8, opt-in) ---
    budget_results = []
    if args.budget_experiment:
        print(f"[F8] seed-budget experiment (budget={config.seed_budget:g}, "
              f"cost_alpha={config.cost_alpha:g}, strategies={list(config.seed_strategies)})")
        budget_results = [
            seed_budget_experiment(config, SimpleContagion(p=0.05)),
            seed_budget_experiment(config, ComplexContagion(phi=0.2)),
        ]
        for br in budget_results:
            viz.fig_budget_cost(
                br, config,
                name=f"F8_budget_cost_{'ic' if br.model_label.startswith('IC') else 'threshold'}",
            )
            print(f"      {br.model_label}: reach winner={br.best_by_reach().strategy}, "
                  f"reach-per-cost winner={br.best_by_reach_per_cost().strategy}")

    # --- persist everything ---
    summary_path = config.results_dir / "summary.json"
    payload = {
        "config": _config_to_jsonable(config),
        "observed_summary": {
            title: summary(sizes) for title, sizes in observed_by_dataset.items()
        },
        "comparisons": [c.to_dict() for c in comparisons],
        "seeding": [s.to_dict() for s in seeding_results],
        "flip_boundary": flip.to_dict(),
        "budget_experiment": [br.to_dict() for br in budget_results],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {summary_path} and figures F1-F5 to {config.results_dir}")


def _config_to_jsonable(config: Config) -> dict:
    out = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        out[f.name] = str(value) if isinstance(value, Path) else value
    return out


if __name__ == "__main__":
    main()
