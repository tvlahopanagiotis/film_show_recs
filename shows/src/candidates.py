"""
Generate candidate TV shows from nearest neighbours.
"""

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
USER_RATINGS = DATA_DIR / "user_ratings.parquet"
SHOWS_META = DATA_DIR / "shows_meta.parquet"
MY_NEIGHBOURS = DATA_DIR / "my_neighbours.parquet"
CANDIDATES_OUT = DATA_DIR / "candidates.parquet"

MIN_RECOMMENDER_COUNT = 3
MIN_AVG_RATING = 7.0          # Trakt 1–10 scale
NEIGHBOUR_LIKE_THRESHOLD = 8.0
TOP_N_CANDIDATES = 60


def norm(slug: str) -> str:
    """Normalise a Trakt slug for deduplication."""
    return re.sub(r"[^a-z0-9]", "", slug.lower())


def _top_20_mean(sims: list) -> float:
    """Mean similarity of top-20% endorsers."""
    if not sims:
        return 0.0
    sorted_sims = sorted(sims, reverse=True)
    top_n = max(1, len(sorted_sims) // 5)
    return sum(sorted_sims[:top_n]) / top_n


def generate_candidates(my_ratings: dict, prefs=None, neighbours=None) -> pd.DataFrame:
    """
    Find shows loved by nearest neighbours that the user hasn't seen.
    Applies session filters from prefs.
    Returns a ranked DataFrame enriched with metadata.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from session import SessionPrefs
    if prefs is None:
        prefs = SessionPrefs()

    t0 = time.time()

    if not MY_NEIGHBOURS.exists():
        raise FileNotFoundError(
            f"{MY_NEIGHBOURS} not found. Run python src/match.py first."
        )

    if neighbours is None:
        neighbours = pd.read_parquet(MY_NEIGHBOURS)
    print(f"Loaded {len(neighbours)} neighbours")

    user_ratings = pd.read_parquet(USER_RATINGS)
    shows_meta = pd.read_parquet(SHOWS_META)

    neighbour_names = set(neighbours["username"].tolist())

    score_col = "combined_score" if "combined_score" in neighbours.columns else "correlation"
    corr_map = dict(zip(neighbours["username"], neighbours[score_col]))

    neighbour_ratings = user_ratings[user_ratings["username"].isin(neighbour_names)].copy()

    my_slugs = set(my_ratings.keys())
    # Import TV_ALREADY_SEEN from match
    from match import TV_ALREADY_SEEN
    already_seen_norms = {norm(s) for s in TV_ALREADY_SEEN}

    liked = neighbour_ratings[
        (neighbour_ratings["rating"] >= NEIGHBOUR_LIKE_THRESHOLD)
        & (~neighbour_ratings["slug"].isin(my_slugs))
    ].copy()

    liked["similarity"] = liked["username"].map(corr_map)
    liked["weighted_score"] = liked["rating"] * liked["similarity"]

    movie_sim_lists = liked.groupby("slug")["similarity"].apply(list)

    agg = liked.groupby("slug").agg(
        weighted_score=("weighted_score", "sum"),
        recommender_count=("username", "nunique"),
        avg_neighbour_rating=("rating", "mean"),
    ).reset_index()

    agg["coverage"] = agg["recommender_count"] / len(neighbours)
    agg["top_neighbour_similarity"] = agg["slug"].map(
        movie_sim_lists.apply(_top_20_mean)
    )

    agg = agg[
        (agg["recommender_count"] >= MIN_RECOMMENDER_COUNT)
        & (agg["avg_neighbour_rating"] >= MIN_AVG_RATING)
    ]

    agg = agg.merge(shows_meta, on="slug", how="left")

    agg["diversity_score"] = (
        agg["weighted_score"] * 0.7
        + agg["avg_neighbour_rating"] * agg["top_neighbour_similarity"] * 0.3
    )

    # Drop rows with no title (incomplete metadata)
    agg = agg[agg["title"].notna() & (agg["title"] != "")]

    # Remove already-seen
    agg["slug_norm"] = agg["slug"].fillna("").apply(norm)
    agg = agg[~agg["slug_norm"].isin(already_seen_norms)]

    # ── Session filters ───────────────────────────────────────────────────────

    # Era filter
    if prefs.era_min_year is not None:
        agg = agg[agg["year"].isna() | (agg["year"] >= prefs.era_min_year)]
    if prefs.era_max_year is not None:
        agg = agg[agg["year"].isna() | (agg["year"] <= prefs.era_max_year)]

    # Status filter
    if prefs.status_filter == "ended":
        agg = agg[agg["status"].isin(["ended", "canceled"])]
    elif prefs.status_filter == "ongoing":
        agg = agg[agg["status"].isin(["returning series", "in production"])]

    # Episode count filter
    if prefs.max_episodes is not None:
        agg = agg[
            agg["aired_episodes"].isna()
            | (agg["aired_episodes"] <= prefs.max_episodes)
        ]

    # Famousness filter (by Trakt vote count)
    if prefs.exclude_famous and "trakt_votes" in agg.columns:
        agg = agg[
            agg["trakt_votes"].isna()
            | (agg["trakt_votes"] < prefs.famous_votes_threshold)
        ]

    agg = agg.sort_values("diversity_score", ascending=False).head(TOP_N_CANDIDATES)
    agg = agg.reset_index(drop=True)

    agg.to_parquet(CANDIDATES_OUT, index=False)
    print(f"Generated {len(agg)} candidates in {time.time()-t0:.1f}s")

    if len(agg) > 0:
        print("\nTop 20 candidates by diversity score:")
        cols = ["title", "recommender_count", "avg_neighbour_rating",
                "top_neighbour_similarity", "diversity_score"]
        available = [c for c in cols if c in agg.columns]
        print(agg.head(20)[available].to_string(index=False))

    return agg
