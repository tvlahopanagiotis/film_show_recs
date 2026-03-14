"""
Load my TV ratings and find nearest neighbours via Pearson correlation.
Run once (or after updating MY_TV_RATINGS): python src/match.py
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

K_NEIGHBOURS = 100
MIN_OVERLAP = 5   # slug overlap required to compute Pearson

# ── My ratings (Trakt slug → 1–10 rating) ───────────────────────────────────
MY_TV_RATINGS: dict[str, float] = {
    "breaking-bad": 10,
    "the-wire": 10,
    "the-sopranos": 10,
    "true-detective": 9,
    "chernobyl": 10,
    "succession": 9,
    "better-call-saul": 9,
    "peaky-blinders": 8,
    "mindhunter": 9,
    "the-queens-gambit": 9,
    "narcos": 8,
    "westworld": 7,
    "money-heist": 8,
    "squid-game": 8,
    "house-of-cards-2013": 8,
    "mad-men": 9,
    "rome": 9,
    "the-big-bang-theory": 6,
    "friends": 7,
    "silicon-valley": 8,
    "brooklyn-nine-nine": 7,
    "bojack-horseman": 9,
    "rick-and-morty": 8,
    "black-mirror": 9,
    "stranger-things": 7,
    "game-of-thrones": 8,
    "the-witcher": 7,
    "suits": 7,
    "killing-eve": 8,
    "lupin": 7,
    "you": 7,
    "the-last-dance": 9,
    "dont-look-up": 7,
    "ted-lasso": 8,
    "two-and-a-half-men": 6,
    "parks-and-recreation": 7,
    "death-note": 8,
}

# Shows to exclude from recommendations (already seen or not interested)
TV_ALREADY_SEEN: set[str] = set(MY_TV_RATINGS.keys()) | {
    "the-mandalorian",
    "the-grand-tour",
    "clarkson-s-farm",
    "the-bear",
    "mr-robot",
    "house-of-cards",           # UK original
    "sherlock",
    "band-of-brothers",
    "planet-earth",
    "planet-earth-ii",
    "the-office-us",
    "how-i-met-your-mother",
    "seinfeld",
    "the-simpsons",
    "curb-your-enthusiasm",
}

# Trakt genre strings (lowercase) — used for genre alignment scoring
PRESTIGE_GENRES = {"drama", "crime", "history", "mystery", "documentary", "thriller", "biography", "war"}
BLOCKBUSTER_GENRES = {"action", "sci-fi", "reality", "animation", "comedy", "fantasy", "horror"}


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
    find_neighbours(MY_TV_RATINGS)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
