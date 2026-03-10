"""
preprocess.py
-------------
Cleans raw data and builds the processed DataFrames used by metrics.py.
Column names match actual CFBD API response (camelCase → snake_case where needed).
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


# ── Games ─────────────────────────────────────────────────────────────────────

def process_games() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "games.csv")

    # CFBD returns camelCase — normalize to snake_case
    df = df.rename(columns={
        "homeTeam":   "home_team",
        "awayTeam":   "away_team",
        "homePoints": "home_points",
        "awayPoints": "away_points",
    })

    uf = df[(df["home_team"] == TEAM) | (df["away_team"] == TEAM)].copy()

    def _won(row):
        if row["home_team"] == TEAM:
            return int(row["home_points"] > row["away_points"])
        return int(row["away_points"] > row["home_points"])

    uf["win"]        = uf.apply(_won, axis=1)
    uf["uf_points"]  = uf.apply(lambda r: r["home_points"] if r["home_team"] == TEAM else r["away_points"], axis=1)
    uf["opp_points"] = uf.apply(lambda r: r["away_points"] if r["home_team"] == TEAM else r["home_points"], axis=1)
    uf["margin"]     = uf["uf_points"] - uf["opp_points"]
    uf["location"]   = uf.apply(lambda r: "home" if r["home_team"] == TEAM else "away", axis=1)

    uf.to_csv(DATA_PROC / "games_clean.csv", index=False)
    print(f"  games_clean      → {len(uf)} rows")
    return uf


# ── Team game stats ───────────────────────────────────────────────────────────

def process_team_stats() -> pd.DataFrame:
    raw   = pd.read_csv(DATA_RAW / "team_game_stats.csv")
    games = pd.read_csv(DATA_PROC / "games_clean.csv")

    # team column may be NaN — filter by game_id instead
    # UF games: keep only the row where home_away matches UF's location in that game
    loc_map = dict(zip(games["id"], games["location"]))  # game_id → "home"/"away"
    win_map = dict(zip(games["id"], games["win"]))

    uf_game_ids = set(games["id"].tolist())
    raw["_uf_location"] = raw["game_id"].map(loc_map)

    # Keep only rows where this team's home_away matches UF's location for that game
    uf = raw[
        (raw["game_id"].isin(uf_game_ids)) &
        (raw["home_away"] == raw["_uf_location"])
    ].copy()
    uf = uf.drop(columns=["_uf_location"])

    # Cast numeric stat columns that exist
    for col in ["rushingYards", "netPassingYards", "totalYards", "turnovers",
                "fumblesLost", "interceptions"]:
        if col in uf.columns:
            uf[col] = pd.to_numeric(uf[col], errors="coerce")

    # Parse efficiency strings e.g. "7-15"
    for eff_col, out_col in [("thirdDownEff", "third_down_pct"),
                              ("fourthDownEff", "fourth_down_pct")]:
        if eff_col in uf.columns:
            uf[out_col] = uf[eff_col].apply(_parse_efficiency)

    # Attach win flag and location
    uf["win"]      = uf["game_id"].map(win_map)
    uf["season"]   = uf["game_id"].map(dict(zip(games["id"], games["season"])))

    uf.to_csv(DATA_PROC / "team_stats_clean.csv", index=False)
    print(f"  team_stats_clean → {len(uf)} rows")
    return uf


# ── Drives ────────────────────────────────────────────────────────────────────

DRIVE_RESULT_MAP = {
    "touchdown":      ["TOUCHDOWN", "TD"],
    "field_goal":     ["FIELD GOAL", "FG"],
    "punt":           ["PUNT"],
    "turnover":       ["FUMBLE", "INTERCEPTION", "TURNOVER"],
    "turnover_downs": ["DOWNS"],
    "end_of_half":    ["END OF HALF", "END OF GAME", "END OF REGULATION"],
}


def _categorize_drive(result: str) -> str:
    r = str(result).upper()
    for category, keywords in DRIVE_RESULT_MAP.items():
        if any(kw in r for kw in keywords):
            return category
    return "other"


def process_drives() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "drives.csv")

    # Filter to UF offense
    if "offense" in df.columns:
        df = df[df["offense"] == TEAM].copy()

    df["plays"] = pd.to_numeric(df["plays"], errors="coerce")
    df["yards"] = pd.to_numeric(df["yards"], errors="coerce")

    # CFBD actual column name is driveResult
    df["result_cat"]  = df["driveResult"].apply(_categorize_drive)
    df["scored"]      = df["result_cat"].isin(["touchdown", "field_goal"])
    df["empty"]       = df["result_cat"].isin(["punt", "turnover", "turnover_downs"])
    df["quick_score"] = df["scored"] & (df["plays"] <= 3)
    df["explosive"]   = df["scored"] & (df["yards"] >= 40)

    df.to_csv(DATA_PROC / "drives_clean.csv", index=False)
    print(f"  drives_clean     → {len(df)} rows")
    return df


# ── 4th downs ─────────────────────────────────────────────────────────────────

def process_fourth_downs() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "fourth_downs.csv")

    # Filter to UF offense
    offense_col = next((c for c in ["offense", "offenseTeam"] if c in df.columns), None)
    if offense_col:
        df = df[df[offense_col] == TEAM].copy()

    # CFBD play columns: playType, playText
    def _label(row):
        pt   = str(row.get("playType", "")).lower()
        desc = str(row.get("playText", "")).lower()
        if "punt" in pt or "punt" in desc:
            return "punt"
        if "field goal" in pt or "field goal" in desc:
            return "fg"
        if any(x in pt for x in ("pass", "rush", "run", "scramble", "kneel")):
            return "go"
        return "other"

    df["decision"] = df.apply(_label, axis=1)
    df.to_csv(DATA_PROC / "fourth_downs_clean.csv", index=False)
    print(f"  fourth_downs_clean → {len(df)} rows")
    return df


# ── SEC season stats ──────────────────────────────────────────────────────────

def process_sec_stats() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "sec_season_stats.csv")

    # Detect column names
    stat_col  = next((c for c in ["statName", "stat"] if c in df.columns), None)
    value_col = next((c for c in ["statValue", "value"] if c in df.columns), None)

    if stat_col is None or value_col is None:
        print(f"  WARNING: SEC stats columns not found. Columns: {df.columns.tolist()}")
        df.to_csv(DATA_PROC / "sec_stats_wide.csv", index=False)
        return df

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
    print(f"  sec_stats_wide   → {len(pivot)} rows")
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
    print(f"\nAll processed files saved to {DATA_PROC}/")
    return out


if __name__ == "__main__":
    process_all()
