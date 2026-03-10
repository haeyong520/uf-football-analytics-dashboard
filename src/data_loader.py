"""
data_loader.py
--------------
All College Football Data API calls.
Each function fetches one endpoint and saves raw CSV to data/raw/.

Usage:
    python src/data_loader.py          # download everything
    from src.data_loader import load_games   # import individual loaders
"""

import time
import requests
import pandas as pd

from config import CFBD_API_KEY, CFBD_BASE_URL, TEAM, SEASONS, DATA_RAW

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}


def _get(endpoint: str, params: dict) -> list:
    """GET with rate-limit pause and clear error messages."""
    url = f"{CFBD_BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    time.sleep(0.35)
    return resp.json()


def load_games(seasons=SEASONS, team=TEAM) -> pd.DataFrame:
    """Game results (scores, home/away, conference flag)."""
    rows = []
    for season in seasons:
        rows.extend(_get("games", {"year": season, "team": team}))
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "games.csv", index=False)
    print(f"  games            → {len(df):>5} rows")
    return df


def load_team_game_stats(seasons=SEASONS, team=TEAM) -> pd.DataFrame:
    """Per-game team stat lines (yards, turnovers, 3rd-down eff., etc.)."""
    rows = []
    for season in seasons:
        data = _get("games/teams", {"year": season, "team": team})
        for game in data:
            game_id = game.get("id")
            for tb in game.get("teams", []):
                flat = {
                    "game_id":   game_id,
                    "season":    season,
                    "team":      tb.get("school"),
                    "home_away": tb.get("homeAway"),
                }
                for stat in tb.get("stats", []):
                    flat[stat["category"]] = stat["stat"]
                rows.append(flat)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "team_game_stats.csv", index=False)
    print(f"  team_game_stats  → {len(df):>5} rows")
    return df


def load_drives(seasons=SEASONS, team=TEAM) -> pd.DataFrame:
    """Drive-level data (result, yards, plays, start/end field position)."""
    rows = []
    for season in seasons:
        data = _get("drives", {"year": season, "team": team})
        for d in data:
            d["season"] = season
        rows.extend(data)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "drives.csv", index=False)
    print(f"  drives           → {len(df):>5} rows")
    return df


def load_fourth_downs(seasons=SEASONS, team=TEAM) -> pd.DataFrame:
    """4th-down play-by-play — loops weeks 1-15 per season."""
    rows = []
    for season in seasons:
        for week in range(1, 16):
            try:
                data = _get("plays", {
                    "year": season, "team": team,
                    "down": 4, "week": week,
                    "seasonType": "regular"
                })
                for p in data:
                    p["season"] = season
                rows.extend(data)
            except Exception:
                pass
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "fourth_downs.csv", index=False)
    print(f"  fourth_downs     → {len(df):>5} rows")
    return df


def load_sec_season_stats(seasons=SEASONS) -> pd.DataFrame:
    """Season-level stats for all SEC teams (used for conference comparison)."""
    rows = []
    for season in seasons:
        data = _get("stats/season", {"year": season, "conference": "SEC"})
        for d in data:
            d["season"] = season
        rows.extend(data)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "sec_season_stats.csv", index=False)
    print(f"  sec_season_stats → {len(df):>5} rows")
    return df


def load_sp_ratings(seasons=SEASONS, team=TEAM) -> pd.DataFrame:
    """SP+ efficiency and explosiveness ratings."""
    rows = []
    for season in seasons:
        data = _get("ratings/sp", {"year": season, "team": team})
        for d in data:
            d["season"] = season
        rows.extend(data)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_RAW / "sp_ratings.csv", index=False)
    print(f"  sp_ratings       → {len(df):>5} rows")
    return df


def load_all() -> None:
    """Download all raw data in one call."""
    print(f"Fetching {TEAM} data for seasons {SEASONS}\n")
    load_games()
    load_team_game_stats()
    load_drives()
    load_fourth_downs()
    load_sec_season_stats()
    load_sp_ratings()
    print(f"\nAll raw data saved to {DATA_RAW}/")


if __name__ == "__main__":
    load_all()
