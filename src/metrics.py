"""
metrics.py
----------
Pure functions that compute analytics metrics from processed DataFrames.
No I/O here — inputs are DataFrames, outputs are DataFrames or scalars.
This separation makes unit-testing straightforward.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import TEAM, SEASONS


# ═══════════════════════════════════════════════════════════════════════════
# TEAM PROFILE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def season_record(games: pd.DataFrame) -> pd.DataFrame:
    """
    Wins, losses, PPG, opponent PPG, and scoring margin per season.

    Parameters
    ----------
    games : processed games_clean.csv

    Returns
    -------
    DataFrame with columns: season, wins, losses, ppg, opp_ppg, margin
    """
    return (
        games.groupby("season")
        .agg(
            wins=("win", "sum"),
            losses=("win", lambda x: len(x) - x.sum()),
            ppg=("uf_points", "mean"),
            opp_ppg=("opp_points", "mean"),
            margin=("margin", "mean"),
        )
        .reset_index()
    )


def per_game_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Season averages: total yards, rush yards, pass yards, turnovers,
    3rd-down conversion rate.

    Parameters
    ----------
    team_stats : processed team_stats_clean.csv

    Returns
    -------
    DataFrame with one row per season.
    """
    agg = {
        "totalYards":      "mean",
        "rushingYards":    "mean",
        "netPassingYards": "mean",
        "turnovers":       "mean",
    }
    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in team_stats.columns}
    if "third_down_pct" in team_stats.columns:
        agg["third_down_pct"] = "mean"

    return team_stats.groupby("season").agg(**{k: (k, v) for k, v in agg.items()}).reset_index()


def home_away_splits(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Mean yards and turnovers split by home vs. away and season.
    """
    cols = [c for c in ["totalYards", "turnovers", "third_down_pct"]
            if c in team_stats.columns]
    return (
        team_stats.groupby(["season", "home_away"])[cols]
        .mean()
        .reset_index()
    )


# ═══════════════════════════════════════════════════════════════════════════
# DRIVE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def drive_summary(drives: pd.DataFrame) -> pd.DataFrame:
    """
    Per-season drive efficiency:
      scoring_pct       — % of drives ending in points
      empty_pct         — % ending in punt / turnover
      yards_per_drive   — mean yards gained per drive
      plays_per_drive   — mean plays per drive
      quick_score_pct   — % of scoring drives completed in ≤ 3 plays
      explosive_pct     — % of scoring drives gaining ≥ 40 yards
    """
    summary = (
        drives.groupby("season")
        .agg(
            n_drives=("scored", "count"),
            scoring_pct=("scored", "mean"),
            empty_pct=("empty", "mean"),
            yards_per_drive=("yards", "mean"),
            plays_per_drive=("plays", "mean"),
            quick_score_pct=("quick_score", "mean"),
            explosive_pct=("explosive", "mean"),
        )
        .reset_index()
    )
    pct_cols = ["scoring_pct", "empty_pct", "quick_score_pct", "explosive_pct"]
    summary[pct_cols] = summary[pct_cols] * 100
    return summary


def drive_result_distribution(drives: pd.DataFrame) -> pd.DataFrame:
    """
    Count and percentage of each drive result category, per season.
    """
    counts = (
        drives.groupby(["season", "result_cat"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("season")["count"].transform("sum")
    counts["pct"] = counts["count"] / totals * 100
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# 4TH DOWN METRICS
# ═══════════════════════════════════════════════════════════════════════════

def fourth_down_decision_rates(fourth: pd.DataFrame) -> pd.DataFrame:
    """
    Go-for-it, punt, and FG rates per season (as percentages).
    """
    counts = (
        fourth.groupby(["season", "decision"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("season")["count"].transform("sum")
    counts["pct"] = counts["count"] / totals * 100
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# WIN vs. LOSS DISCRIMINATORS
# ═══════════════════════════════════════════════════════════════════════════

def win_loss_splits(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Mean value of key metrics in wins vs. losses, per season.
    Useful for identifying the strongest statistical discriminators.
    """
    cols = [c for c in [
        "totalYards", "rushingYards", "netPassingYards",
        "turnovers", "third_down_pct",
    ] if c in team_stats.columns]

    stats = (
        team_stats.dropna(subset=["win"])
        .groupby(["season", "win"])[cols]
        .mean()
        .reset_index()
    )
    stats["result"] = stats["win"].map({1: "Win", 0: "Loss"})
    return stats


def win_loss_effect_sizes(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Cohen's d for each metric comparing wins vs. losses (all seasons pooled).
    Larger absolute value = stronger discriminator.
    """
    cols = [c for c in [
        "totalYards", "rushingYards", "netPassingYards",
        "turnovers", "third_down_pct",
    ] if c in team_stats.columns]

    df = team_stats.dropna(subset=["win"])
    wins   = df[df["win"] == 1]
    losses = df[df["win"] == 0]

    rows = []
    for col in cols:
        w, l = wins[col].dropna(), losses[col].dropna()
        pooled_sd = np.sqrt(((len(w) - 1) * w.std()**2 +
                             (len(l) - 1) * l.std()**2) /
                            (len(w) + len(l) - 2))
        d = (w.mean() - l.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append({
            "metric":        col,
            "mean_wins":     w.mean(),
            "mean_losses":   l.mean(),
            "cohens_d":      round(d, 3),
            "abs_cohens_d":  round(abs(d), 3),
        })

    return pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# SEC CONTEXT METRICS
# ═══════════════════════════════════════════════════════════════════════════

def sec_rankings(sec_wide: pd.DataFrame,
                 season: int,
                 metrics: Optional[list] = None) -> pd.DataFrame:
    """
    Rank all SEC teams on selected metrics for a given season.
    Default metrics: yardsPerPlay, pointsPerGame, thirdDownPct.
    """
    if metrics is None:
        metrics = [c for c in ["yardsPerPlay", "pointsPerGame", "thirdDownPct"]
                   if c in sec_wide.columns]

    sub = sec_wide[sec_wide["season"] == season][["team"] + metrics].copy()

    for m in metrics:
        ascending = m in ("turnoversPerGame",)     # lower = better for these
        sub[f"{m}_rank"] = sub[m].rank(ascending=ascending, method="min")

    return sub.sort_values(f"{metrics[0]}_rank")
