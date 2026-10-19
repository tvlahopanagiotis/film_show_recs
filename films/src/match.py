"""
Load personal ratings and find nearest neighbours via Pearson correlation.
Run once (or after updating my_ratings.csv): python src/match.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_RATINGS = DATA_DIR / "processed_ratings.parquet"
MOVIES_META = DATA_DIR / "movies_meta.parquet"
MY_NEIGHBOURS = DATA_DIR / "my_neighbours.parquet"

MY_RATINGS_PATH = Path(__file__).parent.parent / "my_ratings.csv"

K_NEIGHBOURS = 400
MIN_OVERLAP = 10
ERA_MIN_OVERLAP = 5   # era-only matching has fewer films to share
# Significance weighting: scale correlation by min(overlap, SHRINK_OVERLAP) / SHRINK_OVERLAP,
# so users who happen to agree with me on 10-15 films can't outrank genuine matches.
# Offline evaluation (scripts/evaluate.py): this and dropping the old prestige-genre
# reranking took the neighbour ranking from AUC 0.68 to ~0.85 on its own.
SHRINK_OVERLAP = 50


def load_my_ratings() -> dict:
    """
    Read my_ratings.csv (IMDb export) and return {movieId: my_rating}.
    Bridges IMDb tt-IDs to MovieLens movieIds via movies_meta.parquet.
    """
    ratings_path = MY_RATINGS_PATH
    if not ratings_path.exists():
        fallback = Path(__file__).parent.parent / "imdb_ratings_new.csv"
        if fallback.exists():
            print(f"Note: using {fallback.name} (rename to my_ratings.csv to silence this)")
            ratings_path = fallback
        else:
            raise FileNotFoundError(
                f"Ratings file not found at {MY_RATINGS_PATH}\n"
                "Export your ratings from IMDb and save as my_ratings.csv in the project root."
            )

    my_df = pd.read_csv(ratings_path)

    # Extract numeric IMDb ID: "tt0114709" → 114709
    my_df["imdbId"] = (
        my_df["Const"]
        .str.replace("tt", "", regex=False)
        .str.lstrip("0")
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    my_df = my_df.dropna(subset=["imdbId", "Your Rating"])
    my_df["Your Rating"] = pd.to_numeric(my_df["Your Rating"], errors="coerce")
    my_df = my_df.dropna(subset=["Your Rating"])

    movies_meta = pd.read_parquet(MOVIES_META, columns=["movieId", "imdbId", "title"])
    movies_meta["imdbId"] = movies_meta["imdbId"].astype("Int64")

    merged = my_df.merge(movies_meta, on="imdbId", how="left")
    matched = merged.dropna(subset=["movieId"])
    unmatched = merged[merged["movieId"].isna()]

    print(f"My ratings:  {len(my_df)} total")
    print(f"  Matched:   {len(matched)} to MovieLens IDs ({len(matched)/len(my_df):.0%})")
    print(f"  Unmatched: {len(unmatched)}")
    if not unmatched.empty and "Title" in unmatched.columns:
        titles = unmatched["Title"].tolist()
        preview = titles[:10]
        print(f"  Unmatched titles (first 10): {preview}")
        if len(titles) > 10:
            print(f"  ... and {len(titles) - 10} more")

    result = dict(
        zip(matched["movieId"].astype(int), matched["Your Rating"].astype(float))
    )
    return result


def _pearson_neighbours(ratings: pd.DataFrame, my_ratings: dict, min_overlap: int) -> pd.DataFrame:
    """
    Pearson correlation between my ratings and every user who shares >= min_overlap films,
    shrunk by min(overlap, SHRINK_OVERLAP) / SHRINK_OVERLAP. Returns the top K_NEIGHBOURS.
    """
    overlap_df = ratings[ratings["movieId"].isin(set(my_ratings.keys()))].copy()
    overlap_df["my_rating"] = overlap_df["movieId"].map(my_ratings)

    print(f"  {overlap_df['userId'].nunique():,} users have rated at least one of these films")
    print(f"  Computing Pearson correlation (overlap >= {min_overlap} required)...")

    results = []
    for user_id, group in tqdm(overlap_df.groupby("userId"), desc="Similarity", unit="user"):
        if len(group) < min_overlap:
            continue
        r, _ = pearsonr(group["my_rating"].values, group["rating"].values)
        if not np.isnan(r):
            results.append({
                "userId": int(user_id),
                "correlation": float(r) * min(len(group), SHRINK_OVERLAP) / SHRINK_OVERLAP,
                "raw_correlation": float(r),
                "overlap_count": len(group),
            })

    if not results:
        return pd.DataFrame(columns=["userId", "correlation", "raw_correlation", "overlap_count"])
    return (
        pd.DataFrame(results)
        .sort_values("correlation", ascending=False)
        .head(K_NEIGHBOURS)
        .reset_index(drop=True)
    )


def find_neighbours(my_ratings: dict) -> pd.DataFrame:
    """
    Compute shrunk Pearson correlation between my ratings and every active MovieLens user
    and return the top K_NEIGHBOURS.
    """
    if not PROCESSED_RATINGS.exists():
        raise FileNotFoundError(
            f"{PROCESSED_RATINGS} not found. Run python src/ingest.py first."
        )

    print(f"\nLoading processed ratings...")
    t0 = time.time()
    ratings = pd.read_parquet(PROCESSED_RATINGS)
    print(f"  {len(ratings):,} ratings loaded in {time.time()-t0:.1f}s")

    neighbours = _pearson_neighbours(ratings, my_ratings, MIN_OVERLAP)

    print(f"\nTop 10 neighbours:")
    print(
        neighbours[["userId", "correlation", "raw_correlation", "overlap_count"]]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nMedian overlap across {len(neighbours)} neighbours: {neighbours['overlap_count'].median():.0f} films")

    neighbours.to_parquet(MY_NEIGHBOURS, index=False)
    print(f"\nSaved {len(neighbours)} neighbours to my_neighbours.parquet")
    return neighbours


def get_era_weighted_neighbours(my_ratings: dict, era_min_year: int):
    """
    Recompute neighbours using only my films from era_min_year onwards.
    Returns a neighbours DataFrame in the same format as find_neighbours() —
    session-only, never written to disk.
    Returns None if there aren't enough era-overlapping films to be meaningful.
    """
    ratings = pd.read_parquet(PROCESSED_RATINGS)
    movies_meta = pd.read_parquet(MOVIES_META, columns=["movieId", "year"])

    era_movie_ids = set(
        movies_meta[movies_meta["year"].notna() & (movies_meta["year"] >= era_min_year)]["movieId"].tolist()
    )
    my_era_ratings = {mid: r for mid, r in my_ratings.items() if mid in era_movie_ids}

    if len(my_era_ratings) < MIN_OVERLAP:
        print(
            f"  Warning: only {len(my_era_ratings)} of my rated films are from {era_min_year}+. "
            "Falling back to precomputed neighbours."
        )
        return None

    print(f"\nEra-weighted neighbours: {len(my_era_ratings)} of my films from {era_min_year}+")
    neighbours = _pearson_neighbours(ratings, my_era_ratings, ERA_MIN_OVERLAP)
    if neighbours.empty:
        print("  No neighbours found with sufficient era overlap. Falling back to precomputed neighbours.")
        return None

    print(f"  Top correlation: {neighbours['correlation'].iloc[0]:.3f}")
    return neighbours


if __name__ == "__main__":
    t_start = time.time()
    my_ratings = load_my_ratings()
    find_neighbours(my_ratings)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
