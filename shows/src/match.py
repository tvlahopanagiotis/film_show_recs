"""
Load my TV ratings and find nearest neighbours via Pearson correlation.
Run once (or after replacing films/my_ratings.csv): python src/match.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
USER_RATINGS = DATA_DIR / "user_ratings.parquet"
SHOWS_META = DATA_DIR / "shows_meta.parquet"
MY_NEIGHBOURS = DATA_DIR / "my_neighbours.parquet"
# Same IMDb export the films pipeline reads; TV rows are matched to Trakt by IMDb ID
MY_RATINGS_PATH = Path(__file__).parent.parent.parent / "films" / "my_ratings.csv"
TV_TITLE_TYPES = {"TV Series", "TV Mini Series"}

K_NEIGHBOURS = 100
MIN_OVERLAP = 5   # slug overlap required to compute Pearson

# Shows to exclude from recommendations (seen but not rated, or not interested).
# Rated shows are excluded automatically.
TV_ALREADY_SEEN: set[str] = {
    "planet-earth",
    "planet-earth-ii",
    "the-office-us",
    "how-i-met-your-mother",
    "seinfeld",
    "the-simpsons",
    "curb-your-enthusiasm",
}

# Trakt genre strings (lowercase) — used for genre alignment scoring.
# These must be exact Trakt genre slugs: "science-fiction" (not "sci-fi"), and
# animation is split across "animation" / "anime" / "donghua" with no overlap —
# every anime show is tagged "anime" and NOT "animation".
PRESTIGE_GENRES = {"drama", "crime", "history", "mystery", "documentary", "thriller", "war"}
BLOCKBUSTER_GENRES = {
    "action", "science-fiction", "reality", "comedy", "fantasy", "horror",
    "animation", "anime", "donghua", "children",
}


def load_my_ratings() -> dict:
    """
    Read the TV rows of my_ratings.csv (IMDb export) and return {trakt_slug: my_rating}.
    Bridges IMDb tt-IDs to Trakt slugs via shows_meta.parquet.
    """
    if not MY_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Ratings file not found at {MY_RATINGS_PATH}\n"
            "Export your ratings from IMDb and save as films/my_ratings.csv."
        )

    my_df = pd.read_csv(MY_RATINGS_PATH)
    my_df = my_df[my_df["Title Type"].isin(TV_TITLE_TYPES)].copy()
    my_df["Your Rating"] = pd.to_numeric(my_df["Your Rating"], errors="coerce")
    my_df = my_df.dropna(subset=["Const", "Your Rating"])

    shows_meta = pd.read_parquet(SHOWS_META, columns=["slug", "imdb_id"])
    merged = my_df.merge(shows_meta, left_on="Const", right_on="imdb_id", how="left")
    matched = merged.dropna(subset=["slug"])
    unmatched = merged[merged["slug"].isna()]

    print(f"My TV ratings: {len(my_df)} total")
    print(f"  Matched:     {len(matched)} to Trakt slugs ({len(matched)/max(len(my_df), 1):.0%})")
    if not unmatched.empty:
        print(f"  Unmatched (not in shows_meta): {unmatched['Title'].tolist()}")

    return dict(zip(matched["slug"], matched["Your Rating"].astype(float)))


def compute_genre_alignment(neighbours_df: pd.DataFrame, user_ratings: pd.DataFrame, shows_meta: pd.DataFrame) -> pd.DataFrame:
    """
    For each neighbour, compute genre alignment score.
    +1.0 = pure prestige voter, -1.0 = pure blockbuster voter.
    """
    neighbour_names = set(neighbours_df["username"].tolist())
    neighbour_ratings = user_ratings[
        (user_ratings["username"].isin(neighbour_names))
        & (user_ratings["rating"] >= 8.0)
    ].copy()

    # Trakt genres are stored as a list/array in the parquet — normalise to set of lowercase strings
    meta_genres = shows_meta[["slug", "genres"]].copy()
    neighbour_ratings = neighbour_ratings.merge(meta_genres, on="slug", how="left")

    def _to_genre_set(raw):
        if raw is None:
            return set()
        if hasattr(raw, '__iter__') and not isinstance(raw, str):
            return {str(g).lower() for g in raw}
        return set()

    neighbour_ratings["genre_set"] = neighbour_ratings["genres"].apply(_to_genre_set)

    alignment_scores = {}
    for username, group in neighbour_ratings.groupby("username"):
        good = group["genre_set"].apply(lambda gs: bool(gs & PRESTIGE_GENRES)).sum()
        bad = group["genre_set"].apply(lambda gs: bool(gs & BLOCKBUSTER_GENRES)).sum()
        total = max(good + bad, 1)
        alignment_scores[username] = (good - bad) / total

    neighbours_df = neighbours_df.copy()
    neighbours_df["genre_alignment"] = (
        neighbours_df["username"].map(alignment_scores).fillna(0.0)
    )
    return neighbours_df


def find_neighbours(my_ratings: dict) -> pd.DataFrame:
    """
    Compute Pearson correlation between my ratings and every user in user_ratings.parquet.
    Reranks by genre alignment and returns top K_NEIGHBOURS neighbours.
    """
    if not USER_RATINGS.exists():
        raise FileNotFoundError(
            f"{USER_RATINGS} not found. Run python src/ingest.py first."
        )

    print("\nLoading user ratings...")
    t0 = time.time()
    user_ratings = pd.read_parquet(USER_RATINGS)
    shows_meta = pd.read_parquet(SHOWS_META, columns=["slug", "genres"])
    print(f"  {len(user_ratings):,} ratings from {user_ratings['username'].nunique():,} users in {time.time()-t0:.1f}s")

    my_slugs = set(my_ratings.keys())
    overlap_df = user_ratings[user_ratings["slug"].isin(my_slugs)].copy()
    overlap_df["my_rating"] = overlap_df["slug"].map(my_ratings)

    n_candidate_users = overlap_df["username"].nunique()
    print(f"  {n_candidate_users:,} users have rated at least one of my shows")
    print(f"  Computing Pearson correlation (overlap >= {MIN_OVERLAP} required)...")

    results = []
    grouped = overlap_df.groupby("username")

    for username, group in tqdm(grouped, desc="Similarity", unit="user"):
        if len(group) < MIN_OVERLAP:
            continue
        try:
            r, _ = pearsonr(group["my_rating"].values, group["rating"].values)
            if not np.isnan(r):
                results.append({
                    "username": str(username),
                    "correlation": float(r),
                    "overlap_count": len(group),
                })
        except Exception:
            pass

    neighbours = pd.DataFrame(results).sort_values("correlation", ascending=False)
    neighbours = neighbours.head(K_NEIGHBOURS * 3).reset_index(drop=True)

    print("\nComputing genre alignment scores...")
    neighbours = compute_genre_alignment(neighbours, user_ratings, shows_meta)

    neighbours["combined_score"] = (
        neighbours["correlation"] * 0.6 + neighbours["genre_alignment"] * 0.4
    )
    neighbours = (
        neighbours.sort_values("combined_score", ascending=False)
        .head(K_NEIGHBOURS)
        .reset_index(drop=True)
    )

    print(f"\nTop 10 neighbours after genre alignment reranking:")
    print(
        neighbours[["username", "correlation", "genre_alignment", "combined_score", "overlap_count"]]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nGenre alignment stats across {len(neighbours)} neighbours:")
    print(f"  Mean alignment: {neighbours['genre_alignment'].mean():.3f}")
    print(f"  Prestige-leaning (>0.3): {(neighbours['genre_alignment'] > 0.3).sum()}")
    print(f"  Blockbuster-leaning (<0): {(neighbours['genre_alignment'] < 0).sum()}")

    neighbours.to_parquet(MY_NEIGHBOURS, index=False)
    print(f"\nSaved {len(neighbours)} neighbours to {MY_NEIGHBOURS.name}")
    return neighbours


if __name__ == "__main__":
    t_start = time.time()
    find_neighbours(load_my_ratings())
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
