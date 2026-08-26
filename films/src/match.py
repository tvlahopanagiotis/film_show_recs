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

K_NEIGHBOURS = 200
MIN_OVERLAP = 10


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


def compute_genre_alignment(neighbours_df: pd.DataFrame, processed_ratings: pd.DataFrame, movies_meta: pd.DataFrame) -> pd.DataFrame:
    """
    For each neighbour, compute how much their high-rated films skew toward
    prestige genres vs blockbuster genres.

    Returns neighbours_df with a new 'genre_alignment' column.
    Score ranges from -1.0 (pure blockbuster voter) to +1.0 (pure prestige voter).
    """
    PRESTIGE_GENRES = {"Drama", "Crime", "History", "Biography", "War", "Mystery", "Documentary"}
    BLOCKBUSTER_GENRES = {"Action", "Sci-Fi", "Fantasy", "Horror", "Animation"}
    # Note: Animation sits under BLOCKBUSTER_GENRES both because it attracts
    # mainstream/family voters whose other tastes diverge, and because animation is
    # a genuine negative for me — the few animated titles I rated highly are exceptions.
    # Serious animation (BoJack, Persepolis) still gets a partial offset from the
    # scorer's Animation + Drama combo bonus.

    neighbour_ids = set(neighbours_df["userId"].tolist())
    neighbour_ratings = processed_ratings[
        (processed_ratings["userId"].isin(neighbour_ids))
        & (processed_ratings["rating"] >= 4.0)
    ].copy()

    neighbour_ratings = neighbour_ratings.merge(
        movies_meta[["movieId", "genres"]], on="movieId", how="left"
    )
    neighbour_ratings["genre_set"] = neighbour_ratings["genres"].fillna("").apply(
        lambda g: set(g.split("|"))
    )

    alignment_scores = {}
    for user_id, group in neighbour_ratings.groupby("userId"):
        good = group["genre_set"].apply(lambda gs: bool(gs & PRESTIGE_GENRES)).sum()
        bad = group["genre_set"].apply(lambda gs: bool(gs & BLOCKBUSTER_GENRES)).sum()
        total = max(good + bad, 1)
        alignment_scores[user_id] = (good - bad) / total

    neighbours_df = neighbours_df.copy()
    neighbours_df["genre_alignment"] = (
        neighbours_df["userId"].map(alignment_scores).fillna(0.0)
    )
    return neighbours_df


def find_neighbours(my_ratings: dict) -> pd.DataFrame:
    """
    Compute Pearson correlation between my ratings and every active MovieLens user.
    Reranks by genre alignment and returns top K_NEIGHBOURS neighbours.
    """
    if not PROCESSED_RATINGS.exists():
        raise FileNotFoundError(
            f"{PROCESSED_RATINGS} not found. Run python src/ingest.py first."
        )

    print(f"\nLoading processed ratings...")
    t0 = time.time()
    ratings = pd.read_parquet(PROCESSED_RATINGS)
    print(f"  {len(ratings):,} ratings loaded in {time.time()-t0:.1f}s")

    movies_meta = pd.read_parquet(MOVIES_META, columns=["movieId", "genres"])

    # Filter to only movies I've rated — this massively reduces work
    my_movie_ids = set(my_ratings.keys())
    overlap_df = ratings[ratings["movieId"].isin(my_movie_ids)].copy()
    overlap_df["my_rating"] = overlap_df["movieId"].map(my_ratings)

    n_candidate_users = overlap_df["userId"].nunique()
    print(f"  {n_candidate_users:,} users have rated at least one of my movies")
    print(f"  Computing Pearson correlation (overlap >= {MIN_OVERLAP} required)...")

    results = []
    grouped = overlap_df.groupby("userId")

    for user_id, group in tqdm(grouped, desc="Similarity", unit="user"):
        if len(group) < MIN_OVERLAP:
            continue
        try:
            r, _ = pearsonr(group["my_rating"].values, group["rating"].values)
            if not np.isnan(r):
                results.append(
                    {
                        "userId": int(user_id),
                        "correlation": float(r),
                        "overlap_count": len(group),
                    }
                )
        except Exception:
            pass

    # Take a larger initial pool so genre alignment can promote prestige-leaning
    # neighbours that pure Pearson may have ranked slightly lower
    neighbours = pd.DataFrame(results).sort_values("correlation", ascending=False)
    neighbours = neighbours.head(K_NEIGHBOURS * 3).reset_index(drop=True)

    # Genre alignment reranking
    print("\nComputing genre alignment scores...")
    neighbours = compute_genre_alignment(neighbours, ratings, movies_meta)

    # 60% Pearson (taste similarity) + 40% genre alignment (taste type match)
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
        neighbours[["userId", "correlation", "genre_alignment", "combined_score", "overlap_count"]]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nGenre alignment stats across {len(neighbours)} neighbours:")
    print(f"  Mean alignment: {neighbours['genre_alignment'].mean():.3f}")
    print(f"  Neighbours with alignment > 0.3 (prestige-leaning): {(neighbours['genre_alignment'] > 0.3).sum()}")
    print(f"  Neighbours with alignment < 0 (blockbuster-leaning): {(neighbours['genre_alignment'] < 0).sum()}")

    neighbours.to_parquet(MY_NEIGHBOURS, index=False)
    print(f"\nSaved {len(neighbours)} neighbours to my_neighbours.parquet")
    return neighbours


def get_era_weighted_neighbours(my_ratings: dict, era_min_year: int):
    """
    Recompute Pearson correlation using only films from era_min_year onwards.
    Returns a neighbours DataFrame in the same format as find_neighbours() —
    session-only, never written to disk.
    Returns None if there aren't enough era-overlapping films to be meaningful.
    """
    ratings = pd.read_parquet(PROCESSED_RATINGS)
    movies_meta = pd.read_parquet(MOVIES_META, columns=["movieId", "genres", "year"])

    # Restrict to era films
    era_movie_ids = set(
        movies_meta[movies_meta["year"].notna() & (movies_meta["year"] >= era_min_year)]["movieId"].tolist()
    )

    # My ratings restricted to that era
    my_era_ratings = {mid: r for mid, r in my_ratings.items() if mid in era_movie_ids}

    if len(my_era_ratings) < MIN_OVERLAP:
        print(
            f"  Warning: only {len(my_era_ratings)} of my rated films are from {era_min_year}+. "
            "Falling back to precomputed neighbours."
        )
        return None

    print(f"\nEra-weighted neighbours: {len(my_era_ratings)} of my films from {era_min_year}+")
    print(f"  Computing Pearson on era-only overlap...")

    overlap_df = ratings[ratings["movieId"].isin(set(my_era_ratings.keys()))].copy()
    overlap_df["my_rating"] = overlap_df["movieId"].map(my_era_ratings)

    ERA_MIN_OVERLAP = 3  # lower than global MIN_OVERLAP — era pools are smaller
    results = []
    for user_id, group in tqdm(overlap_df.groupby("userId"), desc="Era similarity", unit="user"):
        if len(group) < ERA_MIN_OVERLAP:
            continue
        try:
            r, _ = pearsonr(group["my_rating"].values, group["rating"].values)
            if not np.isnan(r):
                results.append({
                    "userId": int(user_id),
                    "correlation": float(r),
                    "overlap_count": len(group),
                })
        except Exception:
            pass

    if not results:
        print("  No neighbours found with sufficient era overlap. Falling back to precomputed neighbours.")
        return None

    neighbours = pd.DataFrame(results).sort_values("correlation", ascending=False)
    neighbours = neighbours.head(K_NEIGHBOURS * 3).reset_index(drop=True)

    neighbours = compute_genre_alignment(neighbours, ratings, movies_meta[["movieId", "genres"]])
    neighbours["combined_score"] = (
        neighbours["correlation"] * 0.6 + neighbours["genre_alignment"] * 0.4
    )
    neighbours = (
        neighbours.sort_values("combined_score", ascending=False)
        .head(K_NEIGHBOURS)
        .reset_index(drop=True)
    )

    print(
        f"  Top correlation: {neighbours['correlation'].iloc[0]:.3f}  "
        f"Mean alignment: {neighbours['genre_alignment'].mean():.3f}"
    )
    return neighbours


if __name__ == "__main__":
    t_start = time.time()
    my_ratings = load_my_ratings()
    find_neighbours(my_ratings)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
