"""
viz.py
------
All figure generation functions.
Each plot_*() function accepts a processed DataFrame and saves a PNG.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns

from config import (
    UF_BLUE, UF_ORANGE, SEC_GRAY, COLOR_WIN, COLOR_LOSS,
    FIGURES_DIR, SEASONS,
)

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})


# ── Figure 1 — Season Profile ────────────────────────────────────────────────

def plot_season_profile(record: pd.DataFrame,
                        per_game: pd.DataFrame,
                        fname: str = "fig1_season_profile.png") -> None:
    """2×3 panel: record, scoring margin, yards split, turnover, 3rd-down."""
    merged = record.merge(per_game, on="season", how="left")
    seasons = merged["season"].astype(str).tolist()
    x = np.arange(len(seasons))
    w = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Florida Gators — Season Profile (2022–2024)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Wins / Losses
    ax = axes[0, 0]
    ax.bar(x, merged["wins"],   w, label="Wins",   color=UF_BLUE,   edgecolor="white")
    ax.bar(x, merged["losses"], w, bottom=merged["wins"],
           label="Losses", color=UF_ORANGE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Win-Loss Record"); ax.set_ylabel("Games")
    ax.legend(fontsize=9)

    # Scoring margin
    ax = axes[0, 1]
    ax.plot(seasons, merged["ppg"],     "o-", color=UF_BLUE,   lw=2, ms=8, label="UF PPG")
    ax.plot(seasons, merged["opp_ppg"], "s--", color=UF_ORANGE, lw=2, ms=8, label="Opp PPG")
    ax.fill_between(seasons, merged["ppg"], merged["opp_ppg"], alpha=0.10, color=UF_BLUE)
    ax.set_title("Scoring per Game"); ax.set_ylabel("Points")
    ax.legend(fontsize=9)

    # Rush vs pass yards
    ax = axes[0, 2]
    rkey = "rushingYards" if "rushingYards" in merged.columns else "rush_yards"
    pkey = "netPassingYards" if "netPassingYards" in merged.columns else "pass_yards"
    if rkey in merged.columns and pkey in merged.columns:
        ax.bar(x - w/2, merged[rkey], w, label="Rush", color=UF_BLUE,   edgecolor="white")
        ax.bar(x + w/2, merged[pkey], w, label="Pass", color=UF_ORANGE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Yards/Game — Rush vs. Pass"); ax.set_ylabel("Yards")
    ax.legend(fontsize=9)

    # Scoring margin bars
    ax = axes[1, 0]
    colors = [COLOR_WIN if d >= 0 else COLOR_LOSS for d in merged["margin"]]
    ax.bar(seasons, merged["margin"], color=colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_title("Average Scoring Margin"); ax.set_ylabel("Points/Game")

    # Turnovers
    ax = axes[1, 1]
    tkey = "turnovers" if "turnovers" in merged.columns else "turnovers_pg"
    if tkey in merged.columns:
        ax.bar(seasons, merged[tkey], color=COLOR_LOSS, edgecolor="white", width=0.5)
    ax.set_title("Turnovers per Game"); ax.set_ylabel("Turnovers")
    ax.set_ylim(0, 4)

    # 3rd-down rate
    ax = axes[1, 2]
    tdkey = "third_down_pct"
    if tdkey in merged.columns:
        ax.bar(seasons, merged[tdkey] * 100, color=UF_BLUE, edgecolor="white", width=0.5)
        ax.axhline(42, color="gray", lw=1, ls="--", label="FBS avg ~42%")
    ax.set_title("3rd-Down Conversion %"); ax.set_ylabel("%")
    ax.set_ylim(0, 70); ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, fname)


# ── Figure 2 — Home/Away + 4th Down ─────────────────────────────────────────

def plot_situational(splits: pd.DataFrame,
                     fourth_rates: pd.DataFrame,
                     fname: str = "fig2_situational.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Florida Gators — Situational Football (2022–2024)",
                 fontsize=13, fontweight="bold")

    # Home vs away yards
    ax = axes[0]
    for ha, color, marker in [("home", UF_BLUE, "o"), ("away", UF_ORANGE, "s")]:
        sub = splits[splits["home_away"] == ha]
        col = "totalYards" if "totalYards" in sub.columns else "yards_pg"
        if col in sub.columns:
            ax.plot(sub["season"].astype(str), sub[col],
                    marker=marker, color=color, lw=2, ms=8, label=ha.capitalize())
    ax.set_title("Total Yards/Game: Home vs Away"); ax.set_ylabel("Yards")
    ax.legend(fontsize=9); ax.set_ylim(200, 560)

    # Home vs away turnovers
    ax = axes[1]
    for ha, color, marker in [("home", UF_BLUE, "o"), ("away", UF_ORANGE, "s")]:
        sub = splits[splits["home_away"] == ha]
        if "turnovers" in sub.columns:
            ax.plot(sub["season"].astype(str), sub["turnovers"],
                    marker=marker, color=color, lw=2, ms=8, label=ha.capitalize())
    ax.set_title("Turnovers/Game: Home vs Away"); ax.set_ylabel("Turnovers")
    ax.legend(fontsize=9); ax.set_ylim(0, 4)

    # 4th-down decision stacked bar
    ax = axes[2]
    if len(fourth_rates):
        pivot = fourth_rates.pivot_table(
            index="season", columns="decision", values="pct", aggfunc="sum"
        ).fillna(0)
        color_map = {"punt": UF_ORANGE, "fg": COLOR_WIN, "go": UF_BLUE, "other": SEC_GRAY}
        bottom = np.zeros(len(pivot))
        for col in [c for c in ["punt", "fg", "go", "other"] if c in pivot.columns]:
            ax.bar(pivot.index.astype(str), pivot[col],
                   bottom=bottom, label=col.upper(),
                   color=color_map.get(col, SEC_GRAY), edgecolor="white")
            bottom += pivot[col].values
    ax.set_title("4th-Down Decision Mix (%)"); ax.set_ylabel("%")
    ax.legend(fontsize=9); ax.set_ylim(0, 108)

    plt.tight_layout()
    _save(fig, fname)


# ── Figure 3 — Drive Performance ─────────────────────────────────────────────

def plot_drive_performance(drive_summary: pd.DataFrame,
                           fname: str = "fig3_drive_performance.png") -> None:
    seasons = drive_summary["season"].astype(str).tolist()
    x = np.arange(len(seasons))
    w = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Florida Gators — Drive Performance (2022–2024)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(x, drive_summary["scoring_pct"], 0.5, color=UF_BLUE, edgecolor="white")
    ax.axhline(38, color="gray", lw=1.2, ls="--", label="FBS avg ~38%")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Scoring Drive %"); ax.set_ylabel("%")
    ax.set_ylim(0, 70); ax.legend(fontsize=9)
    for i, v in enumerate(drive_summary["scoring_pct"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    ax = axes[1]
    ax.bar(x, drive_summary["yards_per_drive"], 0.5, color=UF_ORANGE, edgecolor="white")
    ax.axhline(28, color="gray", lw=1.2, ls="--", label="FBS avg ~28 yds")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Yards per Drive"); ax.set_ylabel("Yards")
    ax.set_ylim(0, 50); ax.legend(fontsize=9)
    for i, v in enumerate(drive_summary["yards_per_drive"]):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    ax = axes[2]
    ax.bar(x - w/2, drive_summary["quick_score_pct"], w,
           label="Quick score (≤3 plays)", color=COLOR_WIN, edgecolor="white")
    ax.bar(x + w/2, drive_summary["empty_pct"], w,
           label="Empty drive",            color=COLOR_LOSS, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Quick Scores vs. Empty Drives"); ax.set_ylabel("%")
    ax.set_ylim(0, 60); ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, fname)


# ── Figure 4 — Win vs. Loss ───────────────────────────────────────────────────

def plot_win_loss(wl_splits: pd.DataFrame,
                 effect_sizes: pd.DataFrame,
                 fname: str = "fig4_win_loss.png") -> None:
    metrics = [c for c in [
        "totalYards", "rushingYards", "third_down_pct", "turnovers",
    ] if c in wl_splits.columns]

    labels = {
        "totalYards":      "Total Yards/Game",
        "rushingYards":    "Rush Yards/Game",
        "third_down_pct":  "3rd-Down Conv. %",
        "turnovers":       "Turnovers/Game",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    fig.suptitle("Florida Gators — Wins vs. Losses: Key Stat Splits (2022–2024)",
                 fontsize=13, fontweight="bold")

    if len(metrics) == 1:
        axes = [axes]

    w = 0.35
    for ax, col in zip(axes, metrics):
        wins   = wl_splits[wl_splits["result"] == "Win"]
        losses = wl_splits[wl_splits["result"] == "Loss"]
        x = np.arange(len(SEASONS))
        ax.bar(x - w/2,
               [wins[wins["season"] == s][col].mean()
                if s in wins["season"].values else 0 for s in SEASONS],
               w, label="Win",  color=UF_BLUE,   edgecolor="white")
        ax.bar(x + w/2,
               [losses[losses["season"] == s][col].mean()
                if s in losses["season"].values else 0 for s in SEASONS],
               w, label="Loss", color=UF_ORANGE, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SEASONS])
        ax.set_title(labels.get(col, col))
        ax.legend(fontsize=9)

        # Annotate Cohen's d if available
        row = effect_sizes[effect_sizes["metric"] == col]
        if len(row):
            d = row.iloc[0]["cohens_d"]
            ax.set_xlabel(f"Cohen's d = {d:+.2f}", fontsize=8, color="gray")

    plt.tight_layout()
    _save(fig, fname)


# ── Figure 5 — SEC Context ───────────────────────────────────────────────────

def plot_sec_context(rankings: pd.DataFrame,
                     metrics: list,
                     season: int = 2024,
                     fname: str = "fig5_sec_context.png") -> None:
    n = len(metrics)
    if n == 0:
        print("  No metrics available for SEC context plot.")
        return

    labels = {
        "yardsPerPlay":   "Yards per Play",
        "pointsPerGame":  "Points per Game",
        "thirdDownPct":   "3rd-Down Conv. %",
    }

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Florida vs. SEC — Key Efficiency Metrics ({season})",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes, metrics):
        if col not in rankings.columns:
            continue
        sub = rankings.dropna(subset=[col]).sort_values(col, ascending=True)
        colors = [UF_ORANGE if t == "Florida" else SEC_GRAY
                  for t in sub["team"]]
        ax.barh(sub["team"], sub[col], color=colors, edgecolor="white")
        ax.axvline(sub[col].mean(), color="gray", lw=1, ls="--",
                   label=f"SEC avg ({sub[col].mean():.1f})")
        ax.set_title(labels.get(col, col))
        ax.legend(fontsize=8)

        # Highlight UF bar label
        uf_val = sub[sub["team"] == "Florida"][col].values
        if len(uf_val):
            ax.text(uf_val[0] + 0.05, sub[sub["team"] == "Florida"].index[-1],
                    f"  UF: {uf_val[0]:.1f}", va="center",
                    color=UF_ORANGE, fontsize=8, fontweight="bold")

    plt.tight_layout()
    _save(fig, fname)


# ── Helper ───────────────────────────────────────────────────────────────────

def _save(fig, fname: str) -> None:
    out = FIGURES_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
