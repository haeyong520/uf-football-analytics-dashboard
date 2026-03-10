"""
test_metrics.py
---------------
Unit tests for src/metrics.py.
These run WITHOUT hitting the CFBD API — all inputs are synthetic DataFrames.

Run:
    pytest tests/test_metrics.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

# Patch config so it doesn't raise on missing API key during tests
import unittest.mock as mock
with mock.patch.dict(os.environ, {"CFBD_API_KEY": "test_key"}):
    from metrics import (
        season_record,
        per_game_stats,
        home_away_splits,
        drive_summary,
        drive_result_distribution,
        fourth_down_decision_rates,
        win_loss_splits,
        win_loss_effect_sizes,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_games():
    return pd.DataFrame({
        "season":     [2022, 2022, 2022, 2023, 2023, 2023],
        "win":        [1,    0,    1,    1,    1,    0],
        "uf_points":  [31,   14,   28,   35,   42,   10],
        "opp_points": [14,   21,   17,   21,   14,   24],
        "margin":     [17,   -7,   11,   14,   28,   -14],
    })


@pytest.fixture
def sample_team_stats():
    return pd.DataFrame({
        "season":          [2022, 2022, 2022, 2023, 2023, 2023],
        "home_away":       ["home", "away", "home", "home", "away", "home"],
        "win":             [1,     0,      1,      1,      1,      0],
        "totalYards":      [420,   280,    390,    450,    410,    250],
        "rushingYards":    [180,   100,    160,    200,    190,    90],
        "netPassingYards": [240,   180,    230,    250,    220,    160],
        "turnovers":       [1,     3,      0,      1,      0,      2],
        "third_down_pct":  [0.50,  0.30,   0.45,   0.55,   0.60,   0.25],
    })


@pytest.fixture
def sample_drives():
    return pd.DataFrame({
        "season":      [2022]*6 + [2023]*6,
        "plays":       [5, 3, 8, 2, 6, 4,   4, 2, 7, 3, 5, 6],
        "yards":       [45, 20, 60, 8, 35, 15,  50, 15, 55, 12, 40, 20],
        "scored":      [True,  False, True,  False, True,  False,
                        True,  False, True,  False, True,  False],
        "empty":       [False, True,  False, True,  False, True,
                        False, True,  False, True,  False, True],
        "quick_score": [False, False, True,  False, False, False,
                        False, False, True,  False, False, False],
        "explosive":   [True,  False, True,  False, False, False,
                        True,  False, True,  False, False, False],
        "result_cat":  ["touchdown", "punt", "field_goal", "turnover",
                        "touchdown", "punt",
                        "touchdown", "punt", "field_goal", "turnover",
                        "touchdown", "punt"],
    })


@pytest.fixture
def sample_fourth():
    return pd.DataFrame({
        "season":   [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
        "decision": ["punt", "go", "fg", "punt", "go", "go", "punt", "fg"],
    })


# ── Tests: season_record ──────────────────────────────────────────────────────

def test_season_record_shape(sample_games):
    result = season_record(sample_games)
    assert len(result) == 2                       # 2 seasons
    assert set(result.columns) >= {"season", "wins", "losses", "ppg", "opp_ppg"}


def test_season_record_wins(sample_games):
    result = season_record(sample_games)
    r22 = result[result["season"] == 2022].iloc[0]
    assert r22["wins"] == 2
    assert r22["losses"] == 1


def test_season_record_ppg(sample_games):
    result = season_record(sample_games)
    r22 = result[result["season"] == 2022].iloc[0]
    assert abs(r22["ppg"] - (31 + 14 + 28) / 3) < 0.01


def test_season_record_no_games():
    """Empty DataFrame returns empty result without error."""
    empty = pd.DataFrame(columns=["season", "win", "uf_points", "opp_points", "margin"])
    result = season_record(empty)
    assert len(result) == 0


# ── Tests: drive_summary ──────────────────────────────────────────────────────

def test_drive_summary_pcts_in_range(sample_drives):
    result = drive_summary(sample_drives)
    for col in ["scoring_pct", "empty_pct", "quick_score_pct", "explosive_pct"]:
        assert result[col].between(0, 100).all(), f"{col} out of [0, 100]"


def test_drive_summary_scoring_pct(sample_drives):
    result = drive_summary(sample_drives)
    r22 = result[result["season"] == 2022].iloc[0]
    # 3 scored out of 6 drives = 50%
    assert abs(r22["scoring_pct"] - 50.0) < 0.01


def test_drive_summary_no_nan_for_complete_data(sample_drives):
    result = drive_summary(sample_drives)
    assert result["scoring_pct"].isna().sum() == 0
    assert result["yards_per_drive"].isna().sum() == 0


# ── Tests: fourth_down_decision_rates ────────────────────────────────────────

def test_fourth_down_pct_sums_to_100(sample_fourth):
    result = fourth_down_decision_rates(sample_fourth)
    for season, group in result.groupby("season"):
        total = group["pct"].sum()
        assert abs(total - 100.0) < 0.01, f"Season {season}: pct sums to {total}"


def test_fourth_down_has_expected_decisions(sample_fourth):
    result = fourth_down_decision_rates(sample_fourth)
    decisions = set(result["decision"].unique())
    assert {"punt", "go", "fg"}.issubset(decisions)


# ── Tests: win_loss_effect_sizes ─────────────────────────────────────────────

def test_cohens_d_sign(sample_team_stats):
    """Yards should have positive Cohen's d (wins > losses)."""
    result = win_loss_effect_sizes(sample_team_stats)
    yards_row = result[result["metric"] == "totalYards"].iloc[0]
    assert yards_row["cohens_d"] > 0, "Expected wins to have higher total yards"


def test_cohens_d_turnovers_negative(sample_team_stats):
    """Turnovers should have negative Cohen's d (wins have fewer)."""
    result = win_loss_effect_sizes(sample_team_stats)
    to_row = result[result["metric"] == "turnovers"].iloc[0]
    assert to_row["cohens_d"] < 0, "Expected wins to have fewer turnovers"


def test_effect_size_sorted(sample_team_stats):
    """Results should be sorted by abs Cohen's d descending."""
    result = win_loss_effect_sizes(sample_team_stats)
    vals = result["abs_cohens_d"].tolist()
    assert vals == sorted(vals, reverse=True)


# ── Tests: home_away_splits ──────────────────────────────────────────────────

def test_home_away_splits_has_both_locations(sample_team_stats):
    result = home_away_splits(sample_team_stats)
    locations = set(result["home_away"].unique())
    assert "home" in locations and "away" in locations


def test_home_away_splits_positive_yards(sample_team_stats):
    result = home_away_splits(sample_team_stats)
    assert (result["totalYards"] > 0).all()
