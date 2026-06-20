"""Generate all portfolio figures from outputs/pocket_passes_scored.csv.

No raw tracking/event data required — everything is computed from the scored
candidate table (a few hundred KB), so the figures are fully reproducible from
this repository alone.

A pocket pass is counted when the Stage-2 classifier probability
(``ml_probability``, the LR+GBM ensemble) is >= THRESHOLD (default 0.5).
Team rates are normalised per match played (teams played 3-7 games).

Usage:  python scripts/make_figures.py [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import viz

OUT = ROOT / "outputs"
FIG = OUT / "figures"


def matches_played(df: pd.DataFrame) -> pd.Series:
    played = defaultdict(set)
    for _, r in df[["match_id", "team", "opponent"]].drop_duplicates().iterrows():
        played[r["team"]].add(r["match_id"])
        played[r["opponent"]].add(r["match_id"])
    return pd.Series({t: len(s) for t, s in played.items()})


def _barh(ax, ranked: pd.Series, color, title, xlabel, fmt="{:.1f}"):
    """``ranked`` is in rank order (index 0 = rank #1). Rank #1 is drawn at the TOP."""
    s = ranked.iloc[::-1]                       # reverse so rank #1 lands at top
    ax.barh(s.index, s.values, color=color, height=0.72)
    vmax = max(s.values)
    for y, v in enumerate(s.values):
        ax.text(v + vmax * 0.01, y, fmt.format(v),
                va="center", ha="left", fontsize=9, color=viz.INK)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=viz.MUTED)
    ax.tick_params(length=0)
    ax.set_xlim(0, vmax * 1.12)


def fig_team_rankings(df, mp, thr, top=16):
    hi = df[df["ml_probability"] >= thr]
    attack   = (hi.groupby("team").size() / mp).dropna()
    conceded = (hi.groupby("opponent").size() / mp).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    _barh(axes[0], attack.sort_values(ascending=False).head(top),
          viz.ATTACK, "Most pocket passes PLAYED", "pocket passes per match")
    _barh(axes[1], conceded.sort_values(ascending=True).head(top),
          viz.CONCEDE, "Fewest pocket passes CONCEDED", "pocket passes conceded per match")
    fig.suptitle("Pocket passes into the full-back / centre-back channel — WC2022",
                 fontsize=16, fontweight="bold", x=0.5, y=0.98)
    fig.text(0.5, 0.005,
             f"Stage-2 classifier prob ≥ {thr}; per match played. "
             "Data: PFF FC WC2022 event data (64 matches).",
             ha="center", fontsize=9, color=viz.MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(FIG / "team_rankings.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(df, mp, thr):
    hi = df[df["ml_probability"] >= thr]
    attack   = (hi.groupby("team").size() / mp).dropna()
    conceded = (hi.groupby("opponent").size() / mp).dropna()
    teams = sorted(set(attack.index) & set(conceded.index))
    x = attack[teams]; y = conceded[teams]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(x, y, s=70, color=viz.ATTACK, alpha=0.8, zorder=3, edgecolor="white")
    mx, my = x.mean(), y.mean()
    ax.axvline(mx, color=viz.MUTED, ls="--", lw=1, zorder=1)
    ax.axhline(my, color=viz.MUTED, ls="--", lw=1, zorder=1)
    for t in teams:
        ax.annotate(t, (x[t], y[t]), fontsize=8, xytext=(4, 4),
                    textcoords="offset points", color=viz.INK)
    ax.set_xlabel("Pocket passes PLAYED per match  →  more attacking through the channel",
                  fontsize=11)
    ax.set_ylabel("Pocket passes CONCEDED per match  →  channel more exposed",
                  fontsize=11)
    ax.set_title("Who attacked — and who defended — the FB/CB pocket  (WC2022)",
                 fontsize=15, fontweight="bold", loc="left", pad=12)
    # quadrant hints
    ax.text(ax.get_xlim()[1], my, "  conceded avg", fontsize=8, color=viz.MUTED, va="bottom")
    ax.text(mx, ax.get_ylim()[1], "played avg ", fontsize=8, color=viz.MUTED, ha="right", va="top")
    fig.text(0.5, 0.01, f"Stage-2 classifier prob ≥ {thr}. PFF FC WC2022 event data.",
             ha="center", fontsize=9, color=viz.MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "attack_vs_conceded.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_reception_map(df, thr):
    hi = df[df["ml_probability"] >= thr]
    fig, ax = plt.subplots(figsize=(12, 8))
    viz.draw_pitch(ax)
    hb = ax.hexbin(hi["end_x"], hi["end_y"], gridsize=22, cmap="YlOrRd",
                   mincnt=1, alpha=0.9, zorder=2, extent=(-52.5, 52.5, -34, 34))
    cb = fig.colorbar(hb, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("pocket receptions", fontsize=9)
    cb.ax.tick_params(length=0)
    ax.annotate("attack →", xy=(30, -31), fontsize=11, color=viz.MUTED, style="italic")
    ax.set_title("Where pocket passes are received  (WC2022, all teams)",
                 fontsize=15, fontweight="bold", loc="left")
    fig.text(0.5, 0.02,
             f"Reception point = receiver's location at the next event. "
             f"Classifier prob ≥ {thr}. n = {len(hi)}.",
             ha="center", fontsize=9, color=viz.MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "reception_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_players(df, thr, top=12):
    hi = df[df["ml_probability"] >= thr]
    passers   = hi["passer"].value_counts().head(top)
    receivers = hi["receiver"].value_counts().head(top)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _barh(axes[0], passers.astype(float),
          viz.ATTACK, "Top passers into the pocket", "pocket passes played", fmt="{:.0f}")
    _barh(axes[1], receivers.astype(float),
          "#2E9E6B", "Top receivers in the pocket", "pocket passes received", fmt="{:.0f}")
    fig.suptitle("Individuals — pocket passes  (WC2022)",
                 fontsize=16, fontweight="bold", x=0.5, y=0.99)
    fig.text(0.5, 0.005, f"Stage-2 classifier prob ≥ {thr}. PFF FC WC2022 event data.",
             ha="center", fontsize=9, color=viz.MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(FIG / "top_players.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_combos(df, thr, top=12):
    hi = df[df["ml_probability"] >= thr].copy()
    hi = hi.dropna(subset=["passer", "receiver"])
    hi["combo"] = hi["passer"] + "  →  " + hi["receiver"]
    combos = hi["combo"].value_counts().head(top)

    fig, ax = plt.subplots(figsize=(11, 7))
    _barh(ax, combos.astype(float), "#7B5EA7",
          "Top passer → receiver combinations", "pocket passes", fmt="{:.0f}")
    fig.text(0.5, 0.01, f"Stage-2 classifier prob ≥ {thr}. PFF FC WC2022 event data.",
             ha="center", fontsize=9, color=viz.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "top_combos.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    thr = args.threshold

    viz.apply_style()
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(OUT / "pocket_passes_scored.csv")
    df["match_id"] = df["match_id"].astype(str)
    mp = matches_played(df)

    n_hi = int((df["ml_probability"] >= thr).sum())
    print(f"Scored candidates: {len(df)}  |  prob >= {thr}: {n_hi}  "
          f"|  teams: {len(mp)}  matches: {df['match_id'].nunique()}")

    fig_team_rankings(df, mp, thr)
    fig_scatter(df, mp, thr)
    fig_reception_map(df, thr)
    fig_players(df, thr)
    fig_combos(df, thr)

    print(f"Wrote 5 figures -> {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
