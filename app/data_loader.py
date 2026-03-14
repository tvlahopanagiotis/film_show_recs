"""
Data loading utilities — I/O, path helpers, cached parquet reads.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
FILMS_ROOT = ROOT / "films"
SHOWS_ROOT = ROOT / "shows"
FILMS_DATA = FILMS_ROOT / "data"
SHOWS_DATA = SHOWS_ROOT / "data"


def latest_json(data_dir: Path, prefix: str) -> dict | None:
    """Load the most-recently-dated JSON file matching prefix."""
    files = sorted(data_dir.glob(f"{prefix}*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def parse_genres(raw) -> list[str]:
    """Accept pipe-str (films parquet), numpy array (shows parquet), or list."""
    if isinstance(raw, (list, np.ndarray)):
        return [str(g).strip() for g in raw if g]
    if isinstance(raw, str):
        return [g.strip() for g in raw.split("|") if g.strip()]
    return []


def strip_year(title: str) -> str:
    """Remove trailing year suffix like '(1995)' from film titles."""
    return re.sub(r"\s*\(\d{4}\)$", "", title).strip()


@st.cache_data
def load_my_ratings() -> pd.DataFrame:
    path = FILMS_ROOT / "my_ratings.csv"
    if not path.exists():
        path = FILMS_ROOT / "imdb_ratings_new.csv"
    return pd.read_csv(path)


@st.cache_data
def load_shows_meta() -> pd.DataFrame:
    """For shows Taste Profile — genre lookup per show slug."""
    return pd.read_parquet(SHOWS_DATA / "shows_meta.parquet", columns=["slug", "genres"])


def films_genre_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute avg 'Your Rating' per genre from my_ratings.csv.

    The 'Genres' column is comma-separated, e.g. 'Crime, Drama'.
    """
    rows = []
    for _, row in df.iterrows():
        rating = row.get("Your Rating")
        genres_raw = row.get("Genres", "")
        if pd.isna(rating) or pd.isna(genres_raw):
            continue
        for g in str(genres_raw).split(","):
            g = g.strip()
            if g:
                rows.append({"genre": g, "rating": float(rating)})
    if not rows:
        return {}
    genre_df = pd.DataFrame(rows)
    return genre_df.groupby("genre")["rating"].mean().to_dict()


def shows_genre_stats(my_tv_ratings: dict, shows_meta: pd.DataFrame) -> dict[str, float]:
    """Compute avg rating per genre from MY_TV_RATINGS + shows_meta."""
    rows = []
    for slug, rating in my_tv_ratings.items():
        meta_row = shows_meta[shows_meta["slug"] == slug]
        if meta_row.empty:
            continue
        genres_raw = meta_row.iloc[0]["genres"]
        genres = parse_genres(genres_raw)
        for g in genres:
            rows.append({"genre": g.title(), "rating": float(rating)})
    if not rows:
        return {}
    genre_df = pd.DataFrame(rows)
    return genre_df.groupby("genre")["rating"].mean().to_dict()
