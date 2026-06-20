"""Shared plotting helpers: pitch drawing + a consistent portfolio style."""
from __future__ import annotations

import matplotlib as mpl

# ── portfolio style ─────────────────────────────────────────────────────────────
ATTACK   = "#1A78CF"   # blue  — attacking / playing channel passes
CONCEDE  = "#E1574C"   # red   — defending / conceding channel passes
INK      = "#22303C"   # near-black text
MUTED    = "#8A99A6"   # grey
PITCH_LN = "#C7D0D7"   # pitch lines on white


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.edgecolor":    MUTED,
        "axes.labelcolor":   INK,
        "text.color":        INK,
        "xtick.color":       INK,
        "ytick.color":       INK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def draw_pitch(ax, lw=1.2, color=PITCH_LN):
    """Full pitch, centre-origin metres (105 x 68). Attack is toward +x (right)."""
    from matplotlib.patches import Rectangle, Circle, Arc
    hx, hy = 52.5, 34.0
    ax.add_patch(Rectangle((-hx, -hy), 105, 68, fill=False, color=color, lw=lw))
    ax.plot([0, 0], [-hy, hy], color=color, lw=lw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color=color, lw=lw))
    ax.scatter([0], [0], s=8, color=color, zorder=2)
    for s in (-1, 1):
        # penalty box
        ax.add_patch(Rectangle((s * hx - s * 16.5, -20.16), s * 16.5, 40.32,
                               fill=False, color=color, lw=lw))
        # six-yard box
        ax.add_patch(Rectangle((s * hx - s * 5.5, -9.16), s * 5.5, 18.32,
                               fill=False, color=color, lw=lw))
        # goal
        ax.add_patch(Rectangle((s * hx, -3.66), s * 2.0, 7.32,
                               fill=False, color=color, lw=lw))
        # penalty spot
        ax.scatter([s * (hx - 11)], [0], s=8, color=color, zorder=2)
    ax.set_xlim(-hx - 3, hx + 3)
    ax.set_ylim(-hy - 3, hy + 3)
    ax.set_aspect("equal")
    ax.axis("off")
