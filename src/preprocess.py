"""
preprocess.py
-------------
Cleans raw data and builds the processed DataFrames used by metrics.py.

All functions read from data/raw/ and write to data/processed/.
"""

import pandas as pd
import numpy as np

from config import DATA_RAW, DATA_PROC, TEAM, SEASONS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_efficiency(val: str) -> float:
    """Convert 'n-of-m' strings (e.g. '7-15') to a rate (0.467)."""
    try:
        n, m = str(val).split("-")
        return int(n) / int(m)
    except Exception:
        return np.nan


def _uf_won(row: pd.Series) -> int:
    if row["home_team"] == TEAM:
        return int(row["home_points"] > row["away_points"])
    return int(row["away_points"] > row["home_points"])


def _uf_points(row: pd.Series) -> float:
    return row["home_points"] if row["home_team"] == TEAM else row["away_points"]


def _opp_points(row: pd.Series) -> float:
    return row["away_points"] if row["home_team"] == TEAM else row["home_points"]


# ── Games ─────────────────────────────────────────────────────────────────────

def process_games() -> pd.DataFrame:
    """Add win flag, UF points, opponent points, and game type."""
    df = pd.read_csv(DATA_RAW / "games.csv")

    uf = df[(df["home_team"] == TEAM) | (df["away_team"] == TEAM)].copy()
    uf["win"]        = uf.apply(_uf_won, axis=1)
    uf["uf_points"]  = uf.apply(_uf_points, axis=1)
    uf["opp_points"] = uf.apply(_opp_points, axis=1)
    uf["margin"]     = uf["uf_points"] - uf["opp_points"]
    uf["location"]   = uf.apply(
        lambda r: "home" if r["home_team"] == TEAM else "away", axis=1
    )

    uf.to_csv(DATA_PROC / "games_clean.csv", index=False)
    return uf


# ── Team game stats ───────────────────────────────────────────────────────────

def process_team_stats() -> pd.DataFrame:
    """Cast numeric columns; parse efficiency strings; attach win flag."""
    raw  = pd.read_csv(DATA_RAW / "team_game_stats.csv")
    games = pd.read_csv(DATA_PROC / "games_clean.csv")

    uf = raw[raw["team"] == TEAM].copy()

    numeric_cols = [
        "rushingYards", "netPassingYards", "totalYards",
        "turnovers", "fumblesLost", "interceptions",
    ]
    for col in numeric_cols:
        if col in uf.columns:
            uf[col] = pd.to_numeric(uf[col], errors="coerce")

    for eff_col, out_col in [
        ("thirdDownEff",  "third_down_pct"),
        ("fourthDownEff", "fourth_down_pct"),
    ]:
        if eff_col in uf.columns:
            uf[out_col] = uf[eff_col].apply(_parse_efficiency)

    win_map = dict(zip(games["id"], games["win"]))
    uf["win"] = uf["game_id"].map(win_map)

    uf.to_csv(DATA_PROC / "team_stats_clean.csv", index=False)
    return uf


# ── Drives ────────────────────────────────────────────────────────────────────

DRIVE_RESULT_MAP = {
    "touchdown":        ["TOUCHDOWN", "TD"],
    "field_goal":       ["FIELD GOAL", "FG"],
    "punt":             ["PUNT"],
    "turnover":         ["FUMBLE", "INTERCEPTION", "TURNOVER"],
    "turnover_downs":   ["DOWNS"],
    "end_of_half":      ["END OF HALF", "END OF GAME", "END OF REGULATION"],
}


def _categorize_drive(result: str) -> str:
    r = str(result).upper()
    for category, keywords in DRIVE_RESULT_MAP.items():
        if any(kw in r for kw in keywords):
            return category
    return "other"


def process_drives() -> pd.DataFrame:
    """Add result category, scored flag, and drive efficiency features."""
    df = pd.read_csv(DATA_RAW / "drives.csv")

    if "offense" in df.columns:
        df = df[df["offense"] == TEAM].copy()

    df["plays"] = pd.to_numeric(df.get("plays", 0), errors="coerce")
    df["yards"] = pd.to_numeric(df.get("yards", 0), errors="coerce")

    result_col = "drive_result" if "drive_result" in df.columns else "result"
    df["result_cat"]  = df[result_col].apply(_categorize_drive)
    df["scored"]      = df["result_cat"].isin(["touchdown", "field_goal"])
    df["empty"]       = df["result_cat"].isin(["punt", "turnover", "turnover_downs"])
    df["quick_score"] = df["scored"] & (df["plays"] <= 3)
    df["explosive"]   = df["scored"] & (df["yards"] >= 40)

    df.to_csv(DATA_PROC / "drives_clean.csv", index=False)
    return df


# ── 4th downs ─────────────────────────────────────────────────────────────────

def process_fourth_downs() -> pd.DataFrame:
    """Label each 4th-down play as go / punt / fg / other."""
    df = pd.read_csv(DATA_RAW / "fourth_downs.csv")

    if "offense" in df.columns:
        df = df[df["offense"] == TEAM].copy()

    def _label(row):
        pt   = str(row.get("play_type", "")).lower()
        desc = str(row.get("play_text", row.get("desc", ""))).lower()
        if "punt" in pt or "punt" in desc:
            return "punt"
        if "field goal" in pt or "field goal" in desc:
            return "fg"
        if pt in ("pass", "rush", "run", "qb kneel"):
            return "go"
        return "other"

    df["decision"] = df.apply(_label, axis=1)
    df.to_csv(DATA_PROC / "fourth_downs_clean.csv", index=False)
    return df


# ── SEC season stats ──────────────────────────────────────────────────────────

def process_sec_stats() -> pd.DataFrame:
    """Pivot long-format SEC stats to wide (one row per team-season)."""
    df = pd.read_csv(DATA_RAW / "sec_season_stats.csv")

    stat_col  = "statName"  if "statName"  in df.columns else "stat"
    value_col = "statValue" if "statValue" in df.columns else "value"

    pivot = df.pivot_table(
        index=["season", "team"],
        columns=stat_col,
        values=value_col,
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    for col in pivot.columns[2:]:
        pivot[col] = pd.to_numeric(pivot[col], errors="coerce")

    pivot.to_csv(DATA_PROC / "sec_stats_wide.csv", index=False)
    return pivot


# ── Run all ───────────────────────────────────────────────────────────────────

def process_all() -> dict:
    print("Preprocessing raw data...\n")
    out = {
        "games":        process_games(),
        "team_stats":   process_team_stats(),
        "drives":       process_drives(),
        "fourth_downs": process_fourth_downs(),
        "sec_stats":    process_sec_stats(),
    }
    print(f"\nProcessed files saved to {DATA_PROC}/")
    return out


if __name__ == "__main__":
    process_all()
