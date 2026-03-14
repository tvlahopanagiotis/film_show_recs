"""
Pipeline wiring — calls films/shows pipeline functions with session prefs.

Manages sys.path carefully so films and shows modules don't conflict.
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
FILMS_SRC = ROOT / "films" / "src"
SHOWS_SRC = ROOT / "shows" / "src"
FILMS_DATA = ROOT / "films" / "data"
SHOWS_DATA = ROOT / "shows" / "data"

# Module names that exist in both films/src and shows/src — must be cleared
# from sys.modules when switching between the two pipelines.
_SHARED_MODULE_NAMES = ["session", "candidates", "scorer", "match", "recommend", "ingest"]


def _activate_films() -> None:
    """Ensure films/src is at the front of sys.path; clear stale module cache."""
    for name in _SHARED_MODULE_NAMES:
        sys.modules.pop(name, None)
    films_src = str(FILMS_SRC)
    shows_src = str(SHOWS_SRC)
    sys.path = [p for p in sys.path if p not in (films_src, shows_src)]
    sys.path.insert(0, films_src)


def _activate_shows() -> None:
    """Ensure shows/src is at the front of sys.path; clear stale module cache."""
    for name in _SHARED_MODULE_NAMES:
        sys.modules.pop(name, None)
    films_src = str(FILMS_SRC)
    shows_src = str(SHOWS_SRC)
    sys.path = [p for p in sys.path if p not in (films_src, shows_src)]
    sys.path.insert(0, shows_src)


def run_films(prefs) -> dict:
    """
    Run the fast films pipeline (candidates + scoring; neighbours pre-computed).

    Returns:
        {
            "safe_bets": [...], "wild_cards": [...], "all_scored": [...],
            "n_neighbours": int, "run_time_s": float,
            "session_filters": str, "mode": "films"
        }
    """
    _activate_films()
    from match import load_my_ratings, get_era_weighted_neighbours
    from candidates import generate_candidates
    from scorer import score_candidates
    from recommend import fmt_active_filters

    t0 = time.time()
    my_ratings = load_my_ratings()

    era_neighbours = None
    if prefs.era_min_year is not None:
        era_neighbours = get_era_weighted_neighbours(my_ratings, prefs.era_min_year)

    candidates_df = generate_candidates(my_ratings, prefs, neighbours=era_neighbours)
    results = score_candidates(candidates_df, mood=prefs.mood)

    # Enrich all_scored with diversity_score (not in scorer output)
    div_map = {}
    if "title" in candidates_df.columns and "diversity_score" in candidates_df.columns:
        div_map = dict(zip(candidates_df["title"], candidates_df["diversity_score"].fillna(0)))
    for item in results["all_scored"]:
        item["diversity_score"] = round(float(div_map.get(item["title"], 0.0)), 4)

    results["run_time_s"] = round(time.time() - t0, 2)
    results["session_filters"] = fmt_active_filters(prefs)
    results["n_neighbours"] = len(pd.read_parquet(FILMS_DATA / "my_neighbours.parquet"))
    results["mode"] = "films"
    return results


def run_shows(prefs) -> dict:
    """
    Run the fast shows pipeline (candidates + scoring; neighbours pre-computed).

    Returns:
        {
            "safe_bets": [...], "wild_cards": [...], "all_scored": [...],
            "n_neighbours": int, "run_time_s": float,
            "session_filters": str, "mode": "shows"
        }
    """
    _activate_shows()
    from match import MY_TV_RATINGS
    from candidates import generate_candidates
    from scorer import score_and_select
    from recommend import fmt_active_filters

    t0 = time.time()
    candidates_df = generate_candidates(MY_TV_RATINGS, prefs)
    results = score_and_select(candidates_df, mood=prefs.mood)

    # Enrich all_scored with diversity_score
    div_map = {}
    if "title" in candidates_df.columns and "diversity_score" in candidates_df.columns:
        div_map = dict(zip(candidates_df["title"], candidates_df["diversity_score"].fillna(0)))
    for item in results["all_scored"]:
        item["diversity_score"] = round(float(div_map.get(item["title"], 0.0)), 4)

    results["run_time_s"] = round(time.time() - t0, 2)
    results["session_filters"] = fmt_active_filters(prefs)
    results["n_neighbours"] = len(pd.read_parquet(SHOWS_DATA / "my_neighbours.parquet"))
    results["mode"] = "shows"
    return results


def get_my_tv_ratings() -> dict:
    """Return MY_TV_RATINGS dict from shows/src/match.py."""
    _activate_shows()
    from match import MY_TV_RATINGS
    return MY_TV_RATINGS
