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
    "the-wire": 7,
    "the-sopranos": 8,
    "true-detective": 6,
    "chernobyl": 9,
    "succession": 10,
    "better-call-saul": 8,
    "peaky-blinders": 8,
    "mindhunter": 10,
    "the-queens-gambit": 8,
    "narcos": 8,
    "westworld": 8,
    "money-heist": 4,
    "squid-game": 6,
    "house-of-cards-2013": 8,
    "mad-men": 9,
    "rome": 8,
    "the-big-bang-theory": 1,
    "friends": 8,
    "silicon-valley": 8,
    "brooklyn-nine-nine": 9,
    "bojack-horseman": 9,
    "rick-and-morty": 4,
    "black-mirror": 3,
    "stranger-things": 7,
    "game-of-thrones": 9,
    "the-witcher": 5,
    "suits": 7,
    "killing-eve": 8,
    "lupin": 6,
    "you": 4,
    "the-last-dance": 5,
    "ted-lasso": 8,
    "two-and-a-half-men": 1,
    "parks-and-recreation": 8,
    "death-note": 8,
    # newly rated
    "the-americans": 8,
    "3-body-problem": 7,
    "mare-of-easttown": 7,
    "only-murders-in-the-building": 7,
    "the-night-manager": 7,
    "the-man-in-the-high-castle": 7,
    "the-world-at-war": 9,
    "dexter": 6,
    "fleabag": 2,
    "mr-robot": 8,
    "sherlock": 10,
    "the-bear": 8,
    "clarkson-s-farm": 7,
    "the-grand-tour": 5,
    "the-mandalorian": 6,
    "slow-horses": 7,
    "severance": 7,
}

# Shows to exclude from recommendations (already seen or not interested)
TV_ALREADY_SEEN: set[str] = set(MY_TV_RATINGS.keys()) | {
    "house-of-cards",           # UK original
    "band-of-brothers",
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
