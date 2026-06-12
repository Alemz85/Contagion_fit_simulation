"""Pre-announcement Higgs robustness analysis (DESIGN.md section 5, note 1a).

Why this exists
---------------
In the full Higgs cascade table, the single largest "cascade" holds 87% of all
participation (223,833 of 256,491 nodes) and its next competitor is only 69
nodes. That giant is a *measurement artifact*: around the 4 July 2012
announcement burst, many genuinely separate retweet cascades fuse into one giant
weakly-connected component because they share users. The honest fix is not to
delete the largest point as an "outlier" (the heavy tail is the object of study)
but to remove the *cause*: rebuild the Higgs cascades using only the
pre-announcement window (1-3 July 2012), before that fusion happens.

This script:
    1. rebuilds Higgs cascades from retweets with timestamp < 4 July 2012 UTC
       and writes data/processed/cascades_higgs_pre.csv (cached),
    2. refits IC vs threshold on both substrates (BA and WS),
    3. checks whether the headline 'tie' for full Higgs survives,
    4. writes results/higgs_pre_summary.json and a comparison figure
       results/F7_higgs_pre.{svg,png}.

Run::

    python scripts/higgs_pre_analysis.py --n-nodes 50000 --n-runs 1000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make the src layout importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contagion_fit.config import Config  # noqa: E402
from contagion_fit.data import ccdf, load_cascades, summary  # noqa: E402
from contagion_fit.fit import compare_models  # noqa: E402
from contagion_fit.models import ComplexContagion, SimpleContagion  # noqa: E402
from contagion_fit.simulate import simulate_parallel  # noqa: E402

ACTIVITY = ROOT / "data" / "higgs" / "higgs-activity_time.txt.gz"
PRE_CSV = ROOT / "data" / "processed" / "cascades_higgs_pre.csv"
# 2012-07-04 00:00:00 UTC: start of the announcement day. Events strictly before
# this are the pre-announcement window (1-3 July 2012).
ANNOUNCEMENT_CUTOFF = 1_341_360_000


class _UnionFind:
    """Minimal union-find for weakly-connected components."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        root = x
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def generate_pre_cascades(regen: bool = False) -> None:
    """Build pre-announcement Higgs WCC cascades and cache them to PRE_CSV."""
    if PRE_CSV.exists() and not regen:
        return
    uf = _UnionFind()
    n_rt = 0
    with gzip.open(ACTIVITY, "rt") as f:
        for line in f:
            a, b, ts, kind = line.split()
            if kind != "RT" or int(ts) >= ANNOUNCEMENT_CUTOFF:
                continue
            n_rt += 1
            uf.union(a, b)
    sizes: dict[str, int] = defaultdict(int)
    for node in list(uf.parent):
        sizes[uf.find(node)] += 1
    rows = sorted(sizes.values(), reverse=True)
    with open(PRE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["platform", "cascade_id", "label", "size"])
        for i, s in enumerate(rows):
            w.writerow(["higgs_pre", f"higgs_pre_wcc_{i}", "science_news", s])
    print(f"[generate] pre-announcement RT events: {n_rt:,}  "
          f"cascades: {len(rows):,}  max size: {rows[0]:,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Higgs pre-announcement robustness")
    p.add_argument("--n-nodes", type=int, default=50000)
    p.add_argument("--n-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--regen", action="store_true", help="force-regenerate the subset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_pre_cascades(regen=args.regen)

    full = load_cascades("higgs")
    pre = load_cascades("higgs_pre", path=PRE_CSV)
    print(f"[full higgs] {summary(full)}")
    print(f"[pre  higgs] {summary(pre)}")

    results: dict[str, dict] = {}
    best_sims: dict[str, dict] = {}
    for substrate in ("ba", "ws"):
        config = Config(
            substrate=substrate,
            n_nodes=args.n_nodes,
            n_runs=args.n_runs,
            seed=args.seed,
            n_workers=args.workers,
        )
        cmp = compare_models(config, pre, platform="higgs_pre", label="science_news")
        results[substrate] = cmp.to_dict()
        print(f"[{substrate}] IC {cmp.ic.best_params} d={cmp.ic.best_distance:.3f} | "
              f"Thr {cmp.threshold.best_params} d={cmp.threshold.best_distance:.3f} "
              f"-> {cmp.winner}")
        # Re-simulate at best params for the comparison figure.
        best_sims[substrate] = {
            "ic": simulate_parallel(
                config, SimpleContagion(**cmp.ic.best_params), stream_offset=30_000_000
            ),
            "thr": simulate_parallel(
                config, ComplexContagion(**cmp.threshold.best_params),
                stream_offset=40_000_000,
            ),
            "ic_label": SimpleContagion(**cmp.ic.best_params).label,
            "thr_label": ComplexContagion(**cmp.threshold.best_params).label,
        }

    _write_summary(results, full, pre)
    _make_figure(full, pre, best_sims)
    print("[done] wrote results/higgs_pre_summary.json and results/F7_higgs_pre.{svg,png}")


def _write_summary(results: dict, full: np.ndarray, pre: np.ndarray) -> None:
    payload = {
        "cutoff_unix": ANNOUNCEMENT_CUTOFF,
        "full_higgs_summary": summary(full),
        "pre_higgs_summary": summary(pre),
        "fits": results,
    }
    (ROOT / "results" / "higgs_pre_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _plot_ccdf(ax, sizes, **kw) -> None:
    x, p = ccdf(sizes)
    if x.size:
        ax.loglog(x, p, **kw)


def _make_figure(full: np.ndarray, pre: np.ndarray, best_sims: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: observed full vs pre - the giant tail disappears.
    _plot_ccdf(axes[0], full, color="tab:gray", marker=".", linestyle="none",
               markersize=4, label=f"full higgs (max {full.max():,})")
    _plot_ccdf(axes[0], pre, color="black", marker=".", linestyle="none",
               markersize=4, label=f"pre-announcement (max {pre.max():,})")
    axes[0].set_title("A. Observed: full vs pre-announcement")
    axes[0].legend(fontsize=8)

    # Panels B, C: pre observed vs best fits on each substrate.
    for ax, substrate, name in ((axes[1], "ba", "BA substrate"),
                                (axes[2], "ws", "WS substrate")):
        sim = best_sims[substrate]
        _plot_ccdf(ax, pre, color="black", marker=".", linestyle="none",
                   markersize=4, label="pre observed")
        _plot_ccdf(ax, sim["ic"], color="tab:blue", linestyle="-",
                   label=sim["ic_label"])
        _plot_ccdf(ax, sim["thr"], color="tab:red", linestyle="--",
                   label=sim["thr_label"])
        ax.set_title(f"{'B' if substrate == 'ba' else 'C'}. Pre-Higgs fit ({name})")
        ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("cascade size x")
        ax.set_ylabel("P(X >= x)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle("F7 - Higgs pre-announcement robustness "
                 "(removing the WCC-fusion artifact)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "results" / f"F7_higgs_pre.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
