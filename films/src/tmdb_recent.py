"""
Fetch post-2020 film candidates from TMDB recommendations.

Run after ingest.py whenever you want to refresh recent suggestions:
    python src/tmdb_recent.py           # uses cached result if it exists
    python src/tmdb_recent.py --force   # re-fetches from TMDB

Strategy: take your top-rated films from my_ratings.csv, call TMDB's
recommendations endpoint for each, filter to post-2020, fetch full details
(genres, runtime), and cache to data/tmdb_recent_candidates.parquet.
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
MY_RATINGS_PATH = Path(__file__).parent.parent / "my_ratings.csv"
MOVIES_META = DATA_DIR / "movies_meta.parquet"
TMDB_RECENT_OUT = DATA_DIR / "tmdb_recent_candidates.parquet"

MIN_SEED_RATING = 8   # user's IMDb rating (1–10) to qualify as seed
MAX_SEEDS = 40
RECENT_FROM_YEAR = 2020
MIN_VOTE_COUNT = 50
MIN_VOTE_AVERAGE = 5.5
WORKERS = 8

# TMDB genre names that differ from the scorer's expected names
_GENRE_MAP = {"Science Fiction": "Sci-Fi"}


def _norm_genres(tmdb_genre_names: list[str]) -> str:
    return "|".join(_GENRE_MAP.get(g, g) for g in tmdb_genre_names if g)


def _get_session(tl: threading.local) -> requests.Session:
    if not hasattr(tl, "session"):
        tl.session = requests.Session()
    return tl.session


def fetch_tmdb_recent(force: bool = False) -> pd.DataFrame:
    if TMDB_RECENT_OUT.exists() and not force:
        df = pd.read_parquet(TMDB_RECENT_OUT)
        print(f"Loaded {len(df)} cached TMDB recent candidates (use --force to refresh).")
        return df

    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        print("No TMDB_API_KEY in .env — cannot fetch recent candidates.")
        return pd.DataFrame()

    # ── Seed films: user's top-rated films with known TMDB IDs ───────────────
    ratings_path = MY_RATINGS_PATH
    if not ratings_path.exists():
        ratings_path = Path(__file__).parent.parent / "imdb_ratings_new.csv"

    my_df = pd.read_csv(ratings_path)
    my_df["imdbId"] = (
        my_df["Const"].str.replace("tt", "", regex=False)
        .str.lstrip("0")
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    my_df = my_df.dropna(subset=["imdbId", "Your Rating"])
    my_df["Your Rating"] = pd.to_numeric(my_df["Your Rating"], errors="coerce")

    meta = pd.read_parquet(MOVIES_META, columns=["imdbId", "tmdbId", "title"])
    meta["imdbId"] = meta["imdbId"].astype("Int64")
    merged = my_df.merge(meta, on="imdbId", how="inner").dropna(subset=["tmdbId"])

    seeds = merged[merged["Your Rating"] >= MIN_SEED_RATING].sort_values(
        "Your Rating", ascending=False
    ).head(MAX_SEEDS)
    if len(seeds) < 10:
        seeds = merged.sort_values("Your Rating", ascending=False).head(MAX_SEEDS)

    seed_ids = seeds["tmdbId"].astype(int).tolist()
    print(f"Fetching TMDB recommendations for {len(seed_ids)} seed films rated ≥{MIN_SEED_RATING}★")

    # ── TMDB genre ID → name map (one API call) ───────────────────────────────
    genre_map: dict[int, str] = {}
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/genre/movie/list",
            params={"api_key": api_key},
            timeout=10,
        )
        if r.ok:
            genre_map = {g["id"]: g["name"] for g in r.json().get("genres", [])}
    except Exception:
        pass

    # ── Fetch recommendations for every seed ─────────────────────────────────
    tl = threading.local()

    def _fetch_recs(seed_tmdb_id: int) -> list[dict]:
        results = []
        for page in (1, 2):
            try:
                r = _get_session(tl).get(
                    f"https://api.themoviedb.org/3/movie/{seed_tmdb_id}/recommendations",
                    params={"api_key": api_key, "page": page},
                    timeout=10,
                )
                if r.status_code == 429:
                    time.sleep(2)
                    continue
                if not r.ok:
                    break
                for film in r.json().get("results", []):
                    date = film.get("release_date", "")
                    year = int(date[:4]) if date and len(date) >= 4 else None
                    if not year or year < RECENT_FROM_YEAR:
                        continue
                    if (film.get("vote_count") or 0) < MIN_VOTE_COUNT:
                        continue
                    if (film.get("vote_average") or 0) < MIN_VOTE_AVERAGE:
                        continue
                    results.append({
                        "tmdb_id": film["id"],
                        "title": film.get("title", ""),
                        "year": year,
                        "overview": film.get("overview", ""),
                        "imdb_rating": film.get("vote_average"),
                        "imdb_votes": film.get("vote_count"),
                        "genre_ids": film.get("genre_ids", []),
                    })
            except Exception:
                break
        return results

    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_fetch_recs, tid): tid for tid in seed_ids}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Seed recs", unit="seed"):
            raw.extend(fut.result())

    if not raw:
        print("No post-2020 candidates found from TMDB recommendations.")
        return pd.DataFrame()

    # Deduplicate
    seen: set[int] = set()
    unique = [f for f in raw if not (f["tmdb_id"] in seen or seen.add(f["tmdb_id"]))]
    print(f"Found {len(unique)} unique post-{RECENT_FROM_YEAR} candidates across all seeds")

    # ── Fetch full details for each: genre names + runtime ───────────────────
    def _fetch_details(film: dict) -> dict:
        genres_str = _norm_genres([genre_map.get(gid, "") for gid in film["genre_ids"]])
        runtime = None
        try:
            r = _get_session(tl).get(
                f"https://api.themoviedb.org/3/movie/{film['tmdb_id']}",
                params={"api_key": api_key},
                timeout=10,
            )
            if r.ok:
                data = r.json()
                runtime = data.get("runtime") or None
                full_genres = [g["name"] for g in data.get("genres", [])]
                if full_genres:
                    genres_str = _norm_genres(full_genres)
        except Exception:
            pass

        return {
            "tmdb_id": film["tmdb_id"],
            "title": film["title"],
            "year": film["year"],
            "genres": genres_str,
            "tags_str": film["overview"].lower(),
            "imdb_rating": film["imdb_rating"],
            "imdb_votes": film["imdb_votes"],
            "runtime": runtime,
        }

    details: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_fetch_details, f): f for f in unique}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Details", unit="film"):
            details.append(fut.result())

    df = pd.DataFrame(details)
    df.to_parquet(TMDB_RECENT_OUT, index=False)
    print(f"Saved {len(df)} candidates → {TMDB_RECENT_OUT.name}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = parser.parse_args()
    fetch_tmdb_recent(force=args.force)
