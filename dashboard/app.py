"""
app.py
------
Streamlit dashboard for the UF Football Analytics project.

Run:
    streamlit run dashboard/app.py
"""

import sys
sys.path.insert(0, "src")

import streamlit as st
import pandas as pd
from pathlib import Path

from config import DATA_PROC, FIGURES_DIR, SEASONS, UF_BLUE, UF_ORANGE
from metrics import (
    season_record, per_game_stats, drive_summary,
    win_loss_splits, win_loss_effect_sizes, sec_rankings,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UF Football Analytics",
    page_icon="🐊",
    layout="wide",
)

st.title("🐊 Florida Gators Football Analytics Dashboard")
st.caption("Seasons 2022–2024 | Data: College Football Data API")

# ── Load processed data ───────────────────────────────────────────────────────

@st.cache_data
def load():
    games      = pd.read_csv(DATA_PROC / "games_clean.csv")
    team_stats = pd.read_csv(DATA_PROC / "team_stats_clean.csv")
    drives     = pd.read_csv(DATA_PROC / "drives_clean.csv")
    fourth     = pd.read_csv(DATA_PROC / "fourth_downs_clean.csv")
    sec        = pd.read_csv(DATA_PROC / "sec_stats_wide.csv")
    return games, team_stats, drives, fourth, sec

try:
    games, team_stats, drives, fourth, sec = load()
    data_loaded = True
except FileNotFoundError:
    st.warning(
        "Processed data not found. "
        "Run `python src/data_loader.py` then `python src/preprocess.py` first."
    )
    data_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
season_filter = st.sidebar.multiselect(
    "Season", options=SEASONS, default=SEASONS
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Season Profile",
    "🏟️ Situational",
    "🚗 Drive Performance",
    "📈 Win vs. Loss",
    "🏆 SEC Context",
])

if data_loaded:

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Season-Level Overview")
        g = games[games["season"].isin(season_filter)]
        record = season_record(g)

        col1, col2, col3 = st.columns(3)
        latest = record.iloc[-1]
        col1.metric("2024 Record",
                    f"{int(latest['wins'])}–{int(latest['losses'])}")
        col2.metric("2024 PPG",       f"{latest['ppg']:.1f}")
        col3.metric("2024 Opp PPG",   f"{latest['opp_ppg']:.1f}")

        st.dataframe(record.set_index("season"), use_container_width=True)

        fig_path = FIGURES_DIR / "fig1_season_profile.png"
        if fig_path.exists():
            st.image(str(fig_path))

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Home vs. Away Splits")
        fig_path = FIGURES_DIR / "fig2_situational.png"
        if fig_path.exists():
            st.image(str(fig_path))
        else:
            st.info("Run `python src/viz.py` to generate figures.")

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Drive Efficiency")
        d = drives[drives["season"].isin(season_filter)]
        ds = drive_summary(d)
        st.dataframe(ds.set_index("season").round(1), use_container_width=True)

        fig_path = FIGURES_DIR / "fig3_drive_performance.png"
        if fig_path.exists():
            st.image(str(fig_path))

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("What Separates Wins from Losses?")
        ts = team_stats[team_stats["season"].isin(season_filter)]
        effect = win_loss_effect_sizes(ts)
        st.caption("Cohen's d: positive = wins higher, negative = wins lower")
        st.dataframe(
            effect[["metric", "mean_wins", "mean_losses", "cohens_d"]]
            .set_index("metric").round(3),
            use_container_width=True,
        )

        fig_path = FIGURES_DIR / "fig4_win_loss.png"
        if fig_path.exists():
            st.image(str(fig_path))

    # ── Tab 5 ─────────────────────────────────────────────────────────────────
    with tab5:
        st.subheader("UF vs. SEC (2024)")
        latest_season = max(season_filter) if season_filter else 2024
        avail_metrics = [c for c in ["yardsPerPlay", "pointsPerGame", "thirdDownPct"]
                         if c in sec.columns]
        if avail_metrics:
            rankings = sec_rankings(sec, season=latest_season, metrics=avail_metrics)
            st.dataframe(rankings.set_index("team").round(2),
                         use_container_width=True)

        fig_path = FIGURES_DIR / "fig5_sec_context.png"
        if fig_path.exists():
            st.image(str(fig_path))
