"""
config.py
---------
Central configuration: API credentials, constants, and paths.
API key is loaded from .env — never committed to version control.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
CFBD_API_KEY  = os.getenv("CFBD_API_KEY", "")
CFBD_BASE_URL = "https://api.collegefootballdata.com"

if not CFBD_API_KEY:
    raise EnvironmentError(
        "CFBD_API_KEY not set. "
        "Copy .env.example → .env and add your free key from "
        "https://collegefootballdata.com/key"
    )

# ── Scope ─────────────────────────────────────────────────────────────────────
TEAM    = "Florida"
SEASONS = [2022, 2023, 2024]

SEC_TEAMS = [
    "Florida", "Georgia", "Tennessee", "Alabama", "LSU",
    "Ole Miss", "Mississippi State", "Auburn", "Arkansas",
    "Missouri", "Kentucky", "Vanderbilt", "South Carolina", "Texas A&M",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT_DIR / "data" / "raw"
DATA_PROC   = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR  = ROOT_DIR / "outputs" / "tables"

for _p in [DATA_RAW, DATA_PROC, FIGURES_DIR, TABLES_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Visual identity (UF brand colors) ────────────────────────────────────────
UF_BLUE    = "#003087"
UF_ORANGE  = "#FA4616"
SEC_GRAY   = "#888888"
COLOR_WIN  = "#2ecc71"
COLOR_LOSS = "#e74c3c"
