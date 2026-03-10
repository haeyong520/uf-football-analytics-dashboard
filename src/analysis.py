"""
analysis.py
===========
Four analytical modules for the UF Football Analytics project:

  Module 1 — Team Profile          (season-level summary)
  Module 2 — Situational Football   (down/distance/field position splits)
  Module 3 — Drive Performance      (drive outcome distributions)
  Module 4 — Win vs. Loss Scripts   (game script divergence)

Each module exposes a build_*() function that returns a DataFrame
and a plot_*() function that saves a figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
UF_BLUE   = "#003087"
UF_ORANGE = "#FA4616"
SEC_GRAY  = "#888888"
GOOD      = "#2ecc71"
BAD       = "#e74c3c"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
})

SEASONS = [2022, 2023, 2024]


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1 — TEAM PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def build_team_profile(games_path="data/games.csv",
                       stats_path="data/team_game_stats.csv") -> pd.DataFrame:
    """
    Season-level team profile: record, scoring, yards, turnovers,
    third-down rate, red zone efficiency.
    """
    games = pd.read_csv(games_path)
    stats = pd.read_csv(stats_path)

    # Filter to Florida only
    uf_stats = stats[stats["team"] == "Florida"].copy()

    # Numeric cast
    numeric_cols = ["rushingYards", "netPassingYards", "totalYards",
                    "turnovers", "thirdDownEff", "fourthDownEff",
                    "redZoneScoring", "possessionTime"]
    for col in numeric_cols:
        if col in uf_stats.columns:
            uf_stats[col] = pd.to_numeric(uf_stats[col], errors="coerce")

    # Parse 3rd-down efficiency (e.g. "7-15" → 0.467)
    def parse_eff(val):
        try:
            parts = str(val).split("-")
            return int(parts[0]) / int(parts[1])
        except Exception:
            return np.nan

    if "thirdDownEff" in uf_stats.columns:
        uf_stats["third_down_pct"] = uf_stats["thirdDownEff"].apply(parse_eff)

    # Win/loss from games table
    uf_games = games[
        (games["home_team"] == "Florida") | (games["away_team"] == "Florida")
    ].copy()

    def uf_won(row):
        if row["home_team"] == "Florida":
            return 1 if row["home_points"] > row["away_points"] else 0
        else:
            return 1 if row["away_points"] > row["home_points"] else 0

    uf_games["win"] = uf_games.apply(uf_won, axis=1)
    uf_games["uf_points"] = uf_games.apply(
        lambda r: r["home_points"] if r["home_team"] == "Florida"
                  else r["away_points"], axis=1)
    uf_games["opp_points"] = uf_games.apply(
        lambda r: r["away_points"] if r["home_team"] == "Florida"
                  else r["home_points"], axis=1)

    season_record = uf_games.groupby("season").agg(
        wins=("win", "sum"),
        losses=("win", lambda x: len(x) - x.sum()),
        ppg=("uf_points", "mean"),
        opp_ppg=("opp_points", "mean"),
        point_diff=("uf_points", lambda x:
                    (x - uf_games.loc[x.index, "opp_points"]).mean()),
    ).reset_index()

    per_game = uf_stats.groupby("season").agg(
        yards_per_game=("totalYards", "mean"),
        rush_yards_pg=("rushingYards", "mean"),
        pass_yards_pg=("netPassingYards", "mean"),
        turnovers_pg=("turnovers", "mean"),
        third_down_pct=("third_down_pct", "mean"),
    ).reset_index()

    profile = season_record.merge(per_game, on="season", how="left")
    return profile


def plot_team_profile(profile: pd.DataFrame, out: str):
    """2-row, 3-column season summary panel."""
    seasons = profile["season"].astype(str).tolist()
    x = np.arange(len(seasons))
    w = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Florida Gators Football — Season Profile (2022–2024)",
                 fontsize=15, fontweight="bold", y=1.01)

    # --- Row 1, Col 1: Win-loss record ---
    ax = axes[0, 0]
    ax.bar(x, profile["wins"],   w, label="Wins",   color=UF_BLUE,   edgecolor="white")
    ax.bar(x, profile["losses"], w, bottom=profile["wins"],
           label="Losses", color=UF_ORANGE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Win-Loss Record"); ax.set_ylabel("Games")
    ax.legend(fontsize=9)

    # --- Row 1, Col 2: Scoring margin ---
    ax = axes[0, 1]
    ax.plot(seasons, profile["ppg"],     marker="o", color=UF_BLUE,
            lw=2, ms=8, label="UF PPG")
    ax.plot(seasons, profile["opp_ppg"], marker="s", color=UF_ORANGE,
            lw=2, ms=8, label="Opp PPG", ls="--")
    ax.fill_between(seasons, profile["ppg"], profile["opp_ppg"],
                    alpha=0.12, color=UF_BLUE)
    ax.set_title("Points Per Game"); ax.set_ylabel("Points")
    ax.legend(fontsize=9)

    # --- Row 1, Col 3: Yards per game ---
    ax = axes[0, 2]
    ax.bar(x - w/2, profile["rush_yards_pg"], w,
           label="Rush", color=UF_BLUE,   edgecolor="white")
    ax.bar(x + w/2, profile["pass_yards_pg"], w,
           label="Pass", color=UF_ORANGE, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Yards Per Game (Rush vs. Pass)")
    ax.set_ylabel("Yards"); ax.legend(fontsize=9)

    # --- Row 2, Col 1: Total yards per game ---
    ax = axes[1, 0]
    colors = [UF_BLUE if d >= 0 else UF_ORANGE for d in profile["point_diff"]]
    ax.bar(seasons, profile["point_diff"], color=colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_title("Average Scoring Margin"); ax.set_ylabel("Point Diff per Game")

    # --- Row 2, Col 2: Turnover margin ---
    ax = axes[1, 1]
    ax.bar(seasons, profile["turnovers_pg"], color=BAD, edgecolor="white", width=0.5)
    ax.set_title("Turnovers Per Game"); ax.set_ylabel("Turnovers")
    ax.set_ylim(0, 4)

    # --- Row 2, Col 3: 3rd-down conversion ---
    ax = axes[1, 2]
    ax.bar(seasons, profile["third_down_pct"] * 100,
           color=UF_BLUE, edgecolor="white", width=0.5)
    ax.axhline(42, color="gray", lw=1, ls="--", label="FBS avg (~42%)")
    ax.set_title("3rd-Down Conversion %"); ax.set_ylabel("%")
    ax.set_ylim(0, 70); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2 — SITUATIONAL FOOTBALL
# ═══════════════════════════════════════════════════════════════════════════

def build_situational(stats_path="data/team_game_stats.csv",
                      fourth_path="data/fourth_downs.csv") -> dict:
    """
    Returns dict with:
      'splits'     — home/away splits
      'fourth'     — 4th-down decision breakdown
    """
    stats = pd.read_csv(stats_path)
    fourth = pd.read_csv(fourth_path)

    uf = stats[stats["team"] == "Florida"].copy()
    for col in ["totalYards", "rushingYards", "netPassingYards",
                "thirdDownEff", "turnovers"]:
        if col in uf.columns:
            uf[col] = pd.to_numeric(uf[col], errors="coerce")

    # Home/away splits
    splits = uf.groupby(["season", "home_away"]).agg(
        yards_pg=("totalYards", "mean"),
        turnovers_pg=("turnovers", "mean"),
    ).reset_index()

    # 4th-down decision categories
    fourth_uf = fourth[fourth["offense"] == "Florida"].copy() \
        if "offense" in fourth.columns else fourth.copy()

    def label_4th(row):
        pt = str(row.get("play_type", "")).lower()
        if "punt" in pt:
            return "punt"
        if "field goal" in pt or "fg" in pt:
            return "fg"
        if pt in ("pass", "rush", "run"):
            return "go"
        return "other"

    if len(fourth_uf):
        fourth_uf["decision"] = fourth_uf.apply(label_4th, axis=1)
        fourth_summary = fourth_uf.groupby(
            ["season", "decision"]
        ).size().reset_index(name="count")
    else:
        fourth_summary = pd.DataFrame(
            columns=["season", "decision", "count"]
        )

    return {"splits": splits, "fourth": fourth_summary}


def plot_situational(sit: dict, out: str):
    """Home/away splits + 4th-down decision breakdown."""
    splits = sit["splits"]
    fourth = sit["fourth"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Florida Gators — Situational Football (2022–2024)",
                 fontsize=14, fontweight="bold")

    # -- Home vs Away: Total Yards --
    ax = axes[0]
    for ha, color, marker in [("home", UF_BLUE, "o"), ("away", UF_ORANGE, "s")]:
        sub = splits[splits["home_away"] == ha]
        ax.plot(sub["season"].astype(str), sub["yards_pg"],
                marker=marker, color=color, lw=2, ms=8, label=ha.capitalize())
    ax.set_title("Total Yards/Game: Home vs Away")
    ax.set_ylabel("Yards"); ax.legend(fontsize=9)
    ax.set_ylim(200, 550)

    # -- Home vs Away: Turnovers --
    ax = axes[1]
    for ha, color, marker in [("home", UF_BLUE, "o"), ("away", UF_ORANGE, "s")]:
        sub = splits[splits["home_away"] == ha]
        ax.plot(sub["season"].astype(str), sub["turnovers_pg"],
                marker=marker, color=color, lw=2, ms=8, label=ha.capitalize())
    ax.set_title("Turnovers/Game: Home vs Away")
    ax.set_ylabel("Turnovers"); ax.legend(fontsize=9)
    ax.set_ylim(0, 4)

    # -- 4th-down decision breakdown --
    ax = axes[2]
    if len(fourth):
        pivot = fourth.pivot_table(
            index="season", columns="decision", values="count", aggfunc="sum"
        ).fillna(0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        colors_map = {"go": UF_BLUE, "punt": UF_ORANGE, "fg": GOOD, "other": SEC_GRAY}
        bottom = np.zeros(len(pivot_pct))
        for col in [c for c in ["punt", "fg", "go", "other"] if c in pivot_pct.columns]:
            ax.bar(pivot_pct.index.astype(str), pivot_pct[col],
                   bottom=bottom, label=col.upper(),
                   color=colors_map.get(col, SEC_GRAY), edgecolor="white")
            bottom += pivot_pct[col].values
    ax.set_title("4th-Down Decisions (%)")
    ax.set_ylabel("%"); ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3 — DRIVE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def build_drive_profile(drives_path="data/drives.csv") -> pd.DataFrame:
    """
    Drive-level summary:
      - Points per drive
      - Scoring drive %
      - Empty drive %
      - Average yards per drive
      - Quick-strike drives (≤3 plays, score)
    """
    df = pd.read_csv(drives_path)

    # Filter to Florida offense
    if "offense" in df.columns:
        df = df[df["offense"] == "Florida"].copy()

    df["plays"]  = pd.to_numeric(df.get("plays", 0), errors="coerce")
    df["yards"]  = pd.to_numeric(df.get("yards", 0), errors="coerce")
    df["drive_result"] = df.get("drive_result", df.get("result", "UNKNOWN"))

    scoring_results = {"TD", "FG"}

    def categorize(result):
        r = str(result).upper()
        if "TOUCHDOWN" in r or r == "TD":
            return "touchdown"
        if "FIELD GOAL" in r or r == "FG":
            return "field_goal"
        if "PUNT" in r:
            return "punt"
        if "TURNOVER" in r or "FUMBLE" in r or "INT" in r:
            return "turnover"
        if "DOWNS" in r or "END OF HALF" in r or "END OF GAME" in r:
            return "turnover_on_downs"
        return "other"

    df["result_cat"] = df["drive_result"].apply(categorize)
    df["scored"]     = df["result_cat"].isin(["touchdown", "field_goal"])
    df["empty"]      = df["result_cat"].isin(["punt", "turnover",
                                               "turnover_on_downs"])
    df["quick_score"] = (df["scored"]) & (df["plays"] <= 3)

    summary = df.groupby("season").agg(
        drives=("plays", "count"),
        scoring_pct=("scored", "mean"),
        empty_pct=("empty", "mean"),
        yards_per_drive=("yards", "mean"),
        plays_per_drive=("plays", "mean"),
        quick_score_pct=("quick_score", "mean"),
    ).reset_index()
    summary["scoring_pct"]     *= 100
    summary["empty_pct"]       *= 100
    summary["quick_score_pct"] *= 100

    return summary


def plot_drive_profile(drive_df: pd.DataFrame, out: str):
    """Grouped bar chart of drive outcome categories by season."""
    seasons = drive_df["season"].astype(str).tolist()
    x = np.arange(len(seasons))
    w = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Florida Gators — Drive Performance (2022–2024)",
                 fontsize=14, fontweight="bold")

    # -- Scoring % --
    ax = axes[0]
    ax.bar(x, drive_df["scoring_pct"], width=0.5,
           color=UF_BLUE, edgecolor="white")
    ax.axhline(38, color="gray", lw=1.2, ls="--", label="FBS avg (~38%)")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Scoring Drive %"); ax.set_ylabel("%")
    ax.set_ylim(0, 70); ax.legend(fontsize=9)
    for i, v in enumerate(drive_df["scoring_pct"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    # -- Yards per drive --
    ax = axes[1]
    ax.bar(x, drive_df["yards_per_drive"], width=0.5,
           color=UF_ORANGE, edgecolor="white")
    ax.axhline(28, color="gray", lw=1.2, ls="--", label="FBS avg (~28 yds)")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Yards Per Drive"); ax.set_ylabel("Yards")
    ax.set_ylim(0, 50); ax.legend(fontsize=9)
    for i, v in enumerate(drive_df["yards_per_drive"]):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    # -- Quick score + empty drive --
    ax = axes[2]
    ax.bar(x - w/2, drive_df["quick_score_pct"], w,
           label="Quick score (≤3 plays)", color=GOOD, edgecolor="white")
    ax.bar(x + w/2, drive_df["empty_pct"], w,
           label="Empty drive", color=BAD, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_title("Quick Scores vs. Empty Drives"); ax.set_ylabel("%")
    ax.set_ylim(0, 60); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4 — WIN VS. LOSS GAME SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════

def build_win_loss(games_path="data/games.csv",
                  stats_path="data/team_game_stats.csv") -> pd.DataFrame:
    """
    Compare key stats in wins vs. losses.
    """
    games = pd.read_csv(games_path)
    stats = pd.read_csv(stats_path)

    # Build game-level W/L for UF
    uf_games = games[
        (games["home_team"] == "Florida") | (games["away_team"] == "Florida")
    ].copy()

    def uf_won(row):
        if row["home_team"] == "Florida":
            return int(row["home_points"] > row["away_points"])
        return int(row["away_points"] > row["home_points"])

    uf_games["win"] = uf_games.apply(uf_won, axis=1)
    win_map = dict(zip(uf_games["id"], uf_games["win"]))

    uf_stats = stats[stats["team"] == "Florida"].copy()
    uf_stats["win"] = uf_stats["game_id"].map(win_map)
    uf_stats = uf_stats.dropna(subset=["win"])

    for col in ["totalYards", "rushingYards", "netPassingYards",
                "turnovers", "thirdDownEff"]:
        if col in uf_stats.columns:
            uf_stats[col] = pd.to_numeric(uf_stats[col], errors="coerce")

    def parse_eff(val):
        try:
            parts = str(val).split("-")
            return int(parts[0]) / int(parts[1])
        except Exception:
            return np.nan

    if "thirdDownEff" in uf_stats.columns:
        uf_stats["third_pct"] = uf_stats["thirdDownEff"].apply(parse_eff)

    summary = uf_stats.groupby(["season", "win"]).agg(
        n=("totalYards", "count"),
        total_yards=("totalYards", "mean"),
        rush_yards=("rushingYards", "mean"),
        pass_yards=("netPassingYards", "mean"),
        turnovers=("turnovers", "mean"),
        third_pct=("third_pct", "mean"),
    ).reset_index()

    summary["result"] = summary["win"].map({1: "Win", 0: "Loss"})
    return summary


def plot_win_loss(wl: pd.DataFrame, out: str):
    """Side-by-side bar chart comparing stats in wins vs. losses."""
    metrics = [
        ("total_yards", "Total Yards/Game", ""),
        ("rush_yards",  "Rush Yards/Game",  ""),
        ("third_pct",   "3rd Down Conv. %", "%"),
        ("turnovers",   "Turnovers/Game",   ""),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Florida Gators — Wins vs. Losses: Key Stat Splits (2022–2024)",
                 fontsize=14, fontweight="bold")

    for ax, (col, title, unit) in zip(axes, metrics):
        wins   = wl[wl["result"] == "Win"]
        losses = wl[wl["result"] == "Loss"]
        x = np.arange(len(SEASONS))
        w = 0.35
        ax.bar(x - w/2,
               [wins[wins["season"] == s][col].mean()
                if s in wins["season"].values else 0 for s in SEASONS],
               w, label="Win", color=UF_BLUE, edgecolor="white")
        ax.bar(x + w/2,
               [losses[losses["season"] == s][col].mean()
                if s in losses["season"].values else 0 for s in SEASONS],
               w, label="Loss", color=UF_ORANGE, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SEASONS])
        ax.set_title(title)
        if unit == "%":
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 5 — SEC CONTEXT (BONUS)
# ═══════════════════════════════════════════════════════════════════════════

def build_sec_context(sec_path="data/sec_season_stats.csv") -> pd.DataFrame:
    """
    Compare UF vs SEC peers on key efficiency metrics for most recent season.
    """
    df = pd.read_csv(sec_path)
    # Pivot: one row per team-season, one col per statType
    if "statName" in df.columns:
        df = df.rename(columns={"statName": "stat", "statValue": "value"})
    pivot = df.pivot_table(
        index=["season", "team"], columns="stat", values="value", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    return pivot


def plot_sec_context(sec: pd.DataFrame, out: str, season: int = 2024):
    """
    Dot plot: UF vs SEC average on 3 key metrics for given season.
    """
    sub = sec[sec["season"] == season].copy()
    if sub.empty:
        print(f"  No SEC data for {season}, skipping plot.")
        return

    metrics = []
    for col in ["yardsPerPlay", "pointsPerGame", "thirdDownPct"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            metrics.append(col)

    if not metrics:
        print("  No matching metric columns in SEC data.")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    labels = {
        "yardsPerPlay":   "Yards per Play",
        "pointsPerGame":  "Points per Game",
        "thirdDownPct":   "3rd-Down Conv. %",
    }

    fig.suptitle(f"Florida vs. SEC — Key Efficiency Metrics ({season})",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes, metrics):
        vals = sub[col].dropna()
        ax.scatter(
            vals, sub.loc[vals.index, "team"],
            color=SEC_GRAY, s=60, zorder=3, alpha=0.7
        )
        uf_val = sub[sub["team"] == "Florida"][col].values
        if len(uf_val):
            ax.scatter(
                uf_val[0],
                sub[sub["team"] == "Florida"]["team"].values[0],
                color=UF_ORANGE, s=120, zorder=5,
                edgecolors=UF_BLUE, linewidths=1.5,
                label="Florida"
            )
            ax.axvline(vals.mean(), color="gray", lw=1, ls="--",
                       label=f"SEC avg ({vals.mean():.1f})")
        ax.set_xlabel(labels.get(col, col))
        ax.set_title(labels.get(col, col))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_all(out_dir: str = "figures"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("Module 1 — Team Profile")
    profile = build_team_profile()
    plot_team_profile(profile, f"{out_dir}/fig1_team_profile.png")

    print("Module 2 — Situational Football")
    sit = build_situational()
    plot_situational(sit, f"{out_dir}/fig2_situational.png")

    print("Module 3 — Drive Performance")
    drives = build_drive_profile()
    plot_drive_profile(drives, f"{out_dir}/fig3_drive_performance.png")

    print("Module 4 — Win vs. Loss Scripts")
    wl = build_win_loss()
    plot_win_loss(wl, f"{out_dir}/fig4_win_loss.png")

    print("Module 5 — SEC Context")
    sec = build_sec_context()
    plot_sec_context(sec, f"{out_dir}/fig5_sec_context.png", season=2024)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    run_all()
