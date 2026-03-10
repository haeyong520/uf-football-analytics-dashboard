"""
data_prep.py
============
Downloads Florida Gators football data from the College Football Data API
(https://collegefootballdata.com) and saves locally for analysis.

Usage:
    pip install requests pandas
    python src/data_prep.py

API key: Register free at https://collegefootballdata.com/key
Set as environment variable: export CFBD_API_KEY="your_key_here"
"""

import os
import time
import requests
import pandas as pd

API_KEY  = os.getenv("CFBD_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.collegefootballdata.com"
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

TEAM     = "Florida"
SEASONS  = [2022, 2023, 2024]
OUT_DIR  = "data"

os.makedirs(OUT_DIR, exist_ok=True)


def get(endpoint: str, params: dict) -> list:
    """Generic GET wrapper with error handling."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


# ── 1. Game results ──────────────────────────────────────────────────────────
def fetch_games():
    rows = []
    for season in SEASONS:
        data = get("games", {"year": season, "team": TEAM})
        rows.extend(data)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/games.csv", index=False)
    print(f"  games:      {len(df)} rows")
    return df


# ── 2. Team stats per game ───────────────────────────────────────────────────
def fetch_team_game_stats():
    rows = []
    for season in SEASONS:
        data = get("games/teams", {"year": season, "team": TEAM})
        for game in data:
            game_id = game.get("id")
            for team_block in game.get("teams", []):
                flat = {"game_id": game_id, "season": season,
                        "team": team_block.get("school"),
                        "home_away": team_block.get("homeAway")}
                for stat in team_block.get("stats", []):
                    flat[stat["category"]] = stat["stat"]
                rows.append(flat)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/team_game_stats.csv", index=False)
    print(f"  team_game_stats: {len(df)} rows")
    return df


# ── 3. Drive data ────────────────────────────────────────────────────────────
def fetch_drives():
    rows = []
    for season in SEASONS:
        data = get("drives", {"year": season, "team": TEAM})
        for d in data:
            d["season"] = season
        rows.extend(data)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/drives.csv", index=False)
    print(f"  drives:     {len(df)} rows")
    return df


# ── 4. Play-by-play (4th downs only to keep size manageable) ─────────────────
def fetch_fourth_downs():
    rows = []
    for season in SEASONS:
        data = get("plays", {"year": season, "team": TEAM, "down": 4})
        for p in data:
            p["season"] = season
        rows.extend(data)
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/fourth_downs.csv", index=False)
    print(f"  fourth_downs: {len(df)} rows")
    return df


# ── 5. SEC team stats (for conference comparison) ────────────────────────────
SEC_TEAMS = [
    "Florida", "Georgia", "Tennessee", "Alabama", "LSU",
    "Ole Miss", "Mississippi State", "Auburn", "Arkansas",
    "Missouri", "Kentucky", "Vanderbilt", "South Carolina", "Texas A&M"
]

def fetch_sec_season_stats():
    rows = []
    for season in SEASONS:
        data = get("stats/season", {"year": season, "conference": "SEC"})
        for d in data:
            d["season"] = season
        rows.extend(data)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/sec_season_stats.csv", index=False)
    print(f"  sec_season_stats: {len(df)} rows")
    return df


# ── 6. SP+ ratings (efficiency/explosiveness model) ──────────────────────────
def fetch_sp_ratings():
    rows = []
    for season in SEASONS:
        data = get("ratings/sp", {"year": season, "team": TEAM})
        for d in data:
            d["season"] = season
        rows.extend(data)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/sp_ratings.csv", index=False)
    print(f"  sp_ratings: {len(df)} rows")
    return df


if __name__ == "__main__":
    print(f"Fetching Florida Gators football data for seasons {SEASONS}...\n")
    fetch_games()
    fetch_team_game_stats()
    fetch_drives()
    fetch_fourth_downs()
    fetch_sec_season_stats()
    fetch_sp_ratings()
    print("\nAll data saved to data/")
