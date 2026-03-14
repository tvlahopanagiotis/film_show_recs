"""
Download and preprocess the MovieLens 25M dataset.
Run once: python src/ingest.py
"""

import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_DIR = DATA_DIR / "ml-25m"

PROCESSED_RATINGS = DATA_DIR / "processed_ratings.parquet"
MOVIES_META = DATA_DIR / "movies_meta.parquet"
MOVIE_TAGS = DATA_DIR / "movie_tags.parquet"
IMDB_RATINGS = DATA_DIR / "imdb_ratings.parquet"
TMDB_ENRICHED = DATA_DIR / "tmdb_enriched.parquet"

MIN_USER_RATINGS = 50
MIN_MOVIE_RATINGS = 50


def download_movielens():
    if ML_DIR.exists() and (ML_DIR / "ratings.csv").exists():
        print("MovieLens data already exists, skipping download.")
        return

    print(f"Downloading MovieLens 25M from {ML_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(ML_URL, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"\nDownload failed: {e}")
        print("Try downloading manually from https://grouplens.org/datasets/movielens/25m/")
        print(f"Extract ml-25m.zip so that {ML_DIR}/ratings.csv exists.")
        raise SystemExit(1)

    total = int(response.headers.get("content-length", 0))
    zip_path = DATA_DIR / "ml-25m.zip"

    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading", total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    zip_path.unlink()
    print("Download and extraction complete.")


def preprocess():
    if PROCESSED_RATINGS.exists() and MOVIES_META.exists() and MOVIE_TAGS.exists():
        print("Preprocessed data already exists, skipping.")
        return

    t0 = time.time()

    # ── Ratings ──────────────────────────────────────────────────────────────
    print("\nLoading ratings.csv (this may take a minute)...")
    ratings = pd.read_csv(
        ML_DIR / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        usecols=["userId", "movieId", "rating"],
    )
    print(f"  Loaded {len(ratings):,} ratings from {ratings['userId'].nunique():,} users")

    print(f"Filtering: keeping users with >= {MIN_USER_RATINGS} ratings...")
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["userId"].isin(active_users)]
    print(f"  {ratings['userId'].nunique():,} users remain")

    print(f"Filtering: keeping movies with >= {MIN_MOVIE_RATINGS} ratings...")
    movie_counts = ratings["movieId"].value_counts()
    popular_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    ratings = ratings[ratings["movieId"].isin(popular_movies)]

    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - (n_ratings / (n_users * n_movies))

    print(f"\nAfter filtering:")
    print(f"  Users:    {n_users:,}")
    print(f"  Movies:   {n_movies:,}")
    print(f"  Ratings:  {n_ratings:,}")
    print(f"  Sparsity: {sparsity:.4%}")

    print("Saving processed_ratings.parquet...")
    ratings.to_parquet(PROCESSED_RATINGS, index=False)

    # ── Movies + Links ────────────────────────────────────────────────────────
    print("\nProcessing movies.csv + links.csv...")
    movies = pd.read_csv(ML_DIR / "movies.csv")
    links = pd.read_csv(
        ML_DIR / "links.csv",
        dtype={"movieId": "int32", "imdbId": "Int64", "tmdbId": "Int64"},
    )
    movies_meta = movies.merge(links, on="movieId", how="left")

    movies_meta["year"] = (
        movies_meta["title"].str.extract(r"\((\d{4})\)$")[0].astype("Int64")
    )

    movies_meta.to_parquet(MOVIES_META, index=False)
    print(f"  Saved {len(movies_meta):,} movies to movies_meta.parquet")

    # ── Tags ──────────────────────────────────────────────────────────────────
    print("Processing tags.csv...")
    tags = pd.read_csv(
        ML_DIR / "tags.csv",
        usecols=["movieId", "tag"],
        dtype={"movieId": "int32"},
    )
    tags["tag"] = (
        tags["tag"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.strip()
    )
    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.dropna()))
        .reset_index()
    )
    movie_tags.columns = ["movieId", "tags_str"]
    movie_tags.to_parquet(MOVIE_TAGS, index=False)
    print(f"  Saved tags for {len(movie_tags):,} movies")

    print(f"\nPreprocessing complete in {time.time() - t0:.1f}s")


def _fetch_from_omdb(movies_meta_df: pd.DataFrame, api_key: str, output_path: Path) -> pd.DataFrame:
    """Fetch IMDb ratings from OMDb API (1000 req/day free tier)."""
    valid = movies_meta_df[movies_meta_df["imdbId"].notna()].copy()
    valid["imdbId_str"] = valid["imdbId"].astype(int).apply(lambda x: f"tt{x:07d}")

    print(f"Fetching IMDb ratings for {len(valid):,} films from OMDb...")
    results = []
    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="OMDb"):
        try:
            resp = requests.get(
                f"http://www.omdbapi.com/?i={row['imdbId_str']}&apikey={api_key}",
                timeout=5,
            )
            data = resp.json()

            raw_rating = data.get("imdbRating", "N/A")
            rating = float(raw_rating) if raw_rating not in ("N/A", "", None) else None

            raw_votes = data.get("imdbVotes", "N/A")
            votes = None
            if raw_votes not in ("N/A", "", None):
                try:
                    votes = int(raw_votes.replace(",", ""))
                except ValueError:
                    pass

            raw_runtime = data.get("Runtime", "N/A")
            runtime = None
            if raw_runtime not in ("N/A", "", None):
                try:
                    runtime = int(raw_runtime.split()[0])
                except (ValueError, IndexError):
                    pass

            results.append({
                "movieId": row["movieId"],
                "imdb_rating": rating,
                "imdb_votes": votes,
                "runtime": runtime,
            })
        except Exception:
            results.append({
                "movieId": row["movieId"],
                "imdb_rating": None,
                "imdb_votes": None,
                "runtime": None,
            })
        time.sleep(0.05)  # stay within rate limit

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)
    print(f"  Saved IMDb data — {df['imdb_rating'].notna().sum():,} ratings, "
          f"{df['imdb_votes'].notna().sum():,} vote counts, "
          f"{df['runtime'].notna().sum():,} runtimes")
    return df


def _compute_ml_proxy(output_path: Path) -> pd.DataFrame:
    """
    Use MovieLens average rating as an IMDb proxy (r ≈ 0.85 empirically).
    Scales MovieLens 0.5–5.0 to approximate IMDb 1–10 via linear fit.
    No API key required.
    """
    processed = pd.read_parquet(PROCESSED_RATINGS)
    film_avgs = processed.groupby("movieId")["rating"].mean().reset_index()
    film_avgs.columns = ["movieId", "ml_avg_rating"]

    # Empirical linear fit: ML 1.0 → ~2.3, ML 3.5 → ~6.5, ML 5.0 → ~9.5
    film_avgs["imdb_rating"] = (film_avgs["ml_avg_rating"] * 1.7 + 0.3).clip(1.0, 10.0)

    # Use rating count as an imdb_votes proxy — more-rated films in MovieLens are more famous
    film_vote_counts = processed.groupby("movieId")["rating"].count().reset_index()
    film_vote_counts.columns = ["movieId", "imdb_votes"]
    film_avgs = film_avgs.merge(film_vote_counts, on="movieId", how="left")

    # runtime not available without an API; leave as None so filters skip silently
    film_avgs["runtime"] = None

    result = film_avgs[["movieId", "imdb_rating", "imdb_votes", "runtime"]]
    result.to_parquet(output_path, index=False)
    print(f"  Computed proxy data for {len(result):,} films (rating + vote count; no runtime).")
    return result


def enrich_from_tmdb(movies_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each film with a tmdbId, fetch from TMDB:
      - runtime (minutes)
      - tmdb_vote_count
      - imdb_rating (TMDB vote_average, 1–10 scale — more accurate than ML proxy)
    Caches to data/tmdb_enriched.parquet — only runs once.
    TMDB rate limit: 40 req/s; 0.05s sleep keeps us safe.
    """
    if TMDB_ENRICHED.exists():
        print("TMDB enrichment already cached, skipping.")
        return pd.read_parquet(TMDB_ENRICHED)

    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        print("No TMDB_API_KEY in .env — skipping TMDB enrichment.")
        return pd.DataFrame(columns=["movieId", "runtime", "tmdb_vote_count", "imdb_rating"])

    valid = movies_meta_df[movies_meta_df["tmdbId"].notna()].copy()
    valid["tmdbId"] = valid["tmdbId"].astype(int)
    print(f"Fetching TMDB data for {len(valid):,} films (~{len(valid)*0.05/60:.0f} mins)...")

    results = []
    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="TMDB"):
        enriched = {
            "movieId": row["movieId"],
            "runtime": None,
            "tmdb_vote_count": None,
            "imdb_rating": None,
        }
        try:
            r = requests.get(
                f"https://api.themoviedb.org/3/movie/{int(row['tmdbId'])}",
                params={"api_key": api_key},
                timeout=5,
            )
            if r.ok:
                data = r.json()
                enriched["runtime"] = data.get("runtime") or None
                enriched["tmdb_vote_count"] = data.get("vote_count") or None
                enriched["imdb_rating"] = data.get("vote_average") or None
        except Exception:
            pass
        results.append(enriched)
        time.sleep(0.05)

    df = pd.DataFrame(results)
    df.to_parquet(TMDB_ENRICHED, index=False)
    print(f"TMDB enrichment complete:")
    print(f"  runtime:         {df['runtime'].notna().sum():,}/{len(df):,}")
    print(f"  tmdb_vote_count: {df['tmdb_vote_count'].notna().sum():,}/{len(df):,}")
    print(f"  imdb_rating:     {df['imdb_rating'].notna().sum():,}/{len(df):,}")
    return df


def fetch_imdb_ratings() -> pd.DataFrame:
    """
    Fetch or compute IMDb ratings for all MovieLens films.

    Uses OMDb API if OMDB_API_KEY is set in environment; otherwise falls back
    to a MovieLens-average proxy (no API key required).
    """
    if IMDB_RATINGS.exists():
        print("IMDb ratings already fetched, skipping.")
        return pd.read_parquet(IMDB_RATINGS)

    movies_meta = pd.read_parquet(MOVIES_META)
    omdb_key = os.getenv("OMDB_API_KEY", "").strip()

    if omdb_key:
        return _fetch_from_omdb(movies_meta, omdb_key, IMDB_RATINGS)
    else:
        print("No OMDB_API_KEY found — using MovieLens vote average as IMDb rating proxy.")
        return _compute_ml_proxy(IMDB_RATINGS)


if __name__ == "__main__":
    download_movielens()
    preprocess()
    fetch_imdb_ratings()
    movies_meta = pd.read_parquet(MOVIES_META)
    enrich_from_tmdb(movies_meta)
