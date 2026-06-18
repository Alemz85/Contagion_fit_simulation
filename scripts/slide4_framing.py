"""Slide-4 framing figure: two connection structures x two propagation rules.

Renders a 2x2 "four possible worlds" grid for the presentation:
    columns = propagation rule   (Simple IC  /  Complex threshold)
    rows    = network structure  (Hub-dominated BA / Neighbourhood WS)
Each header carries a tiny schematic so the audience sees the two assumptions
at a glance, and the four cells make the "which world does real data live in?"
question concrete.

Run:  python scripts/slide4_framing.py
Out:  results/slide4_framing.{png,svg}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLUE = "#1f77b4"   # simple / IC  (matches F2)
RED = "#d62728"    # complex / threshold (matches F2)
NODE = "#37474f"   # network nodes
HUB = "#ef6c00"    # hub accent

# Grid geometry.
COL_X = {"simple": 6.7, "complex": 10.2}
ROW_Y = {"hub": 4.0, "nbhd": 1.3}
CELL_W, CELL_H = 3.0, 2.2
CELL_FILL = {
    ("hub", "simple"): "#eef3f8",
    ("hub", "complex"): "#fbeeee",
    ("nbhd", "simple"): "#eef3f8",
    ("nbhd", "complex"): "#fbeeee",
}
WORLD = {("hub", "simple"): "World 1", ("hub", "complex"): "World 2",
         ("nbhd", "simple"): "World 3", ("nbhd", "complex"): "World 4"}


def _dot(ax, x, y, r=0.09, color=NODE, z=3):
    ax.add_patch(Circle((x, y), r, color=color, zorder=z))


def _arrow(ax, p, q, color=NODE, lw=1.6, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, zorder=z, shrinkA=4, shrinkB=4))


def _recruit(ax, xy, color):
    """The node being recruited: hollow (not yet active) -> about to flip."""
    ax.add_patch(Circle(xy, 0.16, facecolor="white", edgecolor=color,
                         lw=2.2, zorder=4))
    ax.text(xy[0], xy[1], "?", ha="center", va="center", fontsize=10,
            color=color, fontweight="bold", zorder=5)


def sketch_simple(ax, cx, cy):
    """One active friend fires across the edge with probability p."""
    src = (cx - 0.85, cy)
    tgt = (cx + 0.7, cy)
    _dot(ax, *src, r=0.12, color=BLUE)            # 1 active friend
    _arrow(ax, src, tgt, color=BLUE)
    ax.text((src[0] + tgt[0]) / 2, cy + 0.26, "prob. p", ha="center",
            va="center", fontsize=10, color=BLUE, fontweight="bold")
    _recruit(ax, tgt, BLUE)                        # you -> activates


def sketch_complex(ax, cx, cy):
    """Fraction phi of neighbours must be active, then adopt with prob. p."""
    srcs = [(cx - 0.95, cy + 0.5), (cx - 0.95, cy), (cx - 0.95, cy - 0.5)]
    tgt = (cx + 0.7, cy)
    for s in srcs:
        _dot(ax, *s, r=0.12, color=RED)            # active friends
        _arrow(ax, s, tgt, color=RED)
    ax.text(cx - 0.3, cy + 0.92, "share >= phi", ha="center", va="center",
            fontsize=10, color=RED, fontweight="bold")
    ax.text(tgt[0] + 0.05, cy + 0.42, "then prob. p", ha="center", va="center",
            fontsize=9, color=RED)
    _recruit(ax, tgt, RED)                         # you -> activates only now


def sketch_hub(ax, cx, cy):
    """Hub-dominated (scale-free): one centre, many spokes."""
    import math
    for k in range(6):
        ang = math.pi / 2 + k * (2 * math.pi / 6)
        lx, ly = cx + 0.62 * math.cos(ang), cy + 0.62 * math.sin(ang)
        ax.plot([cx, lx], [cy, ly], color=NODE, lw=1.0, zorder=1)
        _dot(ax, lx, ly, r=0.08, color=NODE)
    _dot(ax, cx, cy, r=0.17, color=HUB, z=4)


def sketch_nbhd(ax, cx, cy):
    """Neighbourhood (small-world): even degree, dense triangles."""
    pts = [(cx - 0.5, cy + 0.35), (cx + 0.5, cy + 0.35),
           (cx - 0.6, cy - 0.3), (cx + 0.0, cy - 0.5), (cx + 0.6, cy - 0.3)]
    edges = [(0, 1), (0, 2), (1, 4), (2, 3), (3, 4), (0, 3), (1, 3)]
    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=NODE, lw=1.0, zorder=1)
    for p in pts:
        _dot(ax, *p, r=0.08, color=NODE)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title.
    ax.text(6.3, 7.6, "Two assumptions, crossed - four possible worlds",
            ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(6.3, 7.12,
            "How content SPREADS  x  How people CONNECT",
            ha="center", va="center", fontsize=12.5, color="#555555")

    # Column headers (propagation) with sketches.
    head_y = 6.05
    for key, label, sub, sk, col in (
        ("simple", "Simple  (IC)", "each contact fires with prob. p", sketch_simple, BLUE),
        ("complex", "Complex  (threshold)",
         "fraction phi of neighbours, then prob. p", sketch_complex, RED),
    ):
        cx = COL_X[key]
        sk(ax, cx, head_y + 0.15)
        ax.text(cx, head_y - 0.75, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color=col)
        ax.text(cx, head_y - 1.12, sub, ha="center", va="center",
                fontsize=10, color="#666666")
    ax.text(2.4, head_y - 0.1, "PROPAGATION  ->", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#444444")

    # Row headers (network) with sketches.
    row_x = 2.4
    for key, label, sub, sk in (
        ("hub", "Hub-dominated", "Barabasi-Albert", sketch_hub),
        ("nbhd", "Neighbourhood", "Watts-Strogatz", sketch_nbhd),
    ):
        cy = ROW_Y[key]
        sk(ax, row_x, cy + 0.35)
        ax.text(row_x, cy - 0.55, label, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="#333333")
        ax.text(row_x, cy - 0.9, f"({sub})", ha="center", va="center",
                fontsize=9.5, color="#666666")
    ax.text(0.5, 2.65, "NETWORK", ha="center", va="center", rotation=90,
            fontsize=11, fontweight="bold", color="#444444")

    # The four cells.
    for (rk, ck), fill in CELL_FILL.items():
        cx, cy = COL_X[ck], ROW_Y[rk]
        ax.add_patch(FancyBboxPatch(
            (cx - CELL_W / 2, cy - CELL_H / 2), CELL_W, CELL_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.3, edgecolor="#9aa7b1", facecolor=fill, zorder=0))
        ax.text(cx, cy + 0.15, WORLD[(rk, ck)], ha="center", va="center",
                fontsize=15, fontweight="bold", color="#37474f")
        ax.text(cx, cy - 0.45, "= one combination", ha="center", va="center",
                fontsize=9.5, color="#7a8a96")

    # Bottom question.
    ax.text(6.3, 0.0, "Which of the four worlds does real cascade data live in?",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#1a1a1a")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(ROOT / "results" / f"slide4_framing.{ext}",
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote results/slide4_framing.{png,svg}")


if __name__ == "__main__":
    main()
