"""Generate F6: the workflow / flowchart figure (rubric requirement).

DESIGN.md allows a hand-drawn flowchart, but a generated one is reproducible and
matches the visual style of F1-F5. Run as::

    python -m contagion_fit.workflow_figure --results results

This draws the data -> model -> simulate -> fit -> experiments -> figures
pipeline as labelled boxes connected by arrows, and writes F6 as SVG and PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Each box: (x, y, width, height, title, subtitle, colour).
_BOX_W, _BOX_H = 3.1, 1.05
_COL_DATA = "#cfe8ff"
_COL_CORE = "#ffe2b3"
_COL_OUT = "#d7f5d0"

# Layout on a simple grid (column, row) -> centre coordinates.
_BOXES = [
    # (key, col_x, row_y, title, subtitle, colour)
    ("data", 0.0, 6.0, "data.py",
     "load cascades_unified.csv\ncompute observed CCDF", _COL_DATA),
    ("network", 0.0, 4.5, "network.py",
     "build substrate graph\nBA / Watts-Strogatz / Higgs", _COL_DATA),
    ("models", 4.0, 4.5, "models.py",
     "SimpleContagion (IC)\nComplexContagion (threshold)", _COL_CORE),
    ("simulate", 4.0, 6.0, "simulate.py",
     "Monte-Carlo size sampling\nparallel, seeded rng", _COL_CORE),
    ("fit", 8.0, 6.0, "fit.py",
     "grid search + KS distance\nIC vs threshold winner", _COL_CORE),
    ("experiments", 8.0, 4.5, "experiments.py",
     "hub vs random seeding\n(p, phi) flip boundary", _COL_CORE),
    ("viz", 4.0, 3.0, "viz.py",
     "F1 observed CCDF\nF2 best-fit  F3 KS heatmap\nF4 seeding  F5 flip boundary",
     _COL_OUT),
    ("summary", 8.0, 3.0, "summary.json",
     "best params, distances,\nwinners, experiments", _COL_OUT),
]

# Arrows between box keys.
_ARROWS = [
    ("data", "simulate"),
    ("network", "models"),
    ("models", "simulate"),
    ("simulate", "fit"),
    ("simulate", "experiments"),
    ("data", "fit"),
    ("fit", "viz"),
    ("fit", "summary"),
    ("experiments", "viz"),
    ("experiments", "summary"),
]


def _centres() -> dict[str, tuple[float, float]]:
    return {key: (x, y) for key, x, y, *_ in _BOXES}


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))
    centres = _centres()

    for _key, x, y, title, subtitle, colour in _BOXES:
        box = mpatches.FancyBboxPatch(
            (x - _BOX_W / 2, y - _BOX_H / 2), _BOX_W, _BOX_H,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2, edgecolor="#333333", facecolor=colour,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.22, title, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(x, y - 0.18, subtitle, ha="center", va="center", fontsize=8)

    for src, dst in _ARROWS:
        x0, y0 = centres[src]
        x1, y1 = centres[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4,
                            shrinkA=42, shrinkB=42, connectionstyle="arc3,rad=0.05"),
        )

    # Legend for the three stages.
    legend_handles = [
        mpatches.Patch(facecolor=_COL_DATA, edgecolor="#333", label="data / substrate"),
        mpatches.Patch(facecolor=_COL_CORE, edgecolor="#333", label="core engine"),
        mpatches.Patch(facecolor=_COL_OUT, edgecolor="#333", label="outputs"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    ax.set_title("F6 - contagion-fit analysis workflow", fontsize=14, pad=14)
    ax.set_xlim(-2.2, 10.2)
    ax.set_ylim(2.0, 7.0)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="generate F6 workflow figure")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for ext in ("svg", "png"):
        path = args.results / f"F6_workflow.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
