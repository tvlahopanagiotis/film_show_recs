"""
Ingest TV show ratings from Trakt.tv via public API.

Strategy:
  1. Seed from my top-rated shows — find users who rated them highly.
  2. Collect all their show ratings.
  3. Fetch metadata for every show that appears.

Run once: python src/ingest.py
Requires TRAKT_CLIENT_ID in .env (free at trakt.tv/oauth/applications).
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

USER_RATINGS_OUT = DATA_DIR / "user_ratings.parquet"
SHOWS_META_OUT = DATA_DIR / "shows_meta.parquet"

TRAKT_API_KEY = os.getenv("TRAKT_CLIENT_ID", "").strip()
TRAKT_BASE = "https://api.trakt.tv"

# Shows I've rated highly — used to find like-minded users
SEED_SLUGS = [
    "breaking-bad",
    "the-wire",
    "the-sopranos",
    "true-detective",
    "chernobyl",
    "succession",
    "better-call-saul",
    "peaky-blinders",
    "mindhunter",
    "the-queens-gambit",
    "narcos",
    "westworld",
    "money-heist",
    "squid-game",
    "house-of-cards-2013",
    "mad-men",
    "rome",
    "black-mirror",
    "bojack-horseman",
    "the-last-dance",
    "game-of-thrones",
    "killing-eve",
    "silicon-valley",
    "ted-lasso",
    "death-note",
]

MIN_SEED_OVERLAP = 4   # user must have rated at least this many seed shows
MAX_USERS = 300        # number of neighbours to collect ratings for
REQUEST_SLEEP = 0.3    # seconds between API requests (well within rate limit)


def trakt_get(path: str, params: dict = None) -> dict | list | None:
    """GET a Trakt API endpoint. Returns parsed JSON or None on error."""
    if not TRAKT_API_KEY:
        raise RuntimeError(
            "TRAKT_CLIENT_ID not set. Add it to shows/.env\n"
            "Register a free app at https://trakt.tv/oauth/applications"
        )
    headers = {
        "Content-Type": "application/json",
        "trakt-api-version": "2",
        "trakt-api-key": TRAKT_API_KEY,
    }
    try:
        r = requests.get(
            f"{TRAKT_BASE}{path}",
            headers=headers,
            params=params or {},
            timeout=10,
        )
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 10))
            print(f"  Rate limited — sleeping {retry_after}s")
            time.sleep(retry_after)
            return trakt_get(path, params)
        if not r.ok:
            return None
        return r.json()
    except Exception:
        return None


def fetch_show_raters(slug: str, limit: int = 200) -> list[str]:
    """
    Return up to `limit` usernames who have rated this show.
    Uses GET /shows/{slug}/ratings — returns rating distribution but not users.

    Since Trakt doesn't expose a raters-list endpoint on the free tier,
    we instead fetch the show's comment/shout thread to get engaged users,
    and supplement with watchers.
    """
    # Fetch users who checked in / watched via /shows/{slug}/watching (live)
    # Best available proxy: /shows/{slug}/comments returns usernames of commenters
    usernames = set()

    # Approach: use /movies/{slug}/lists to find lists containing the show
    # (lists are public and include the username of the list owner)
    data = trakt_get(f"/shows/{slug}/lists/personal/added", params={"limit": limit})
    if data:
        for item in data:
            username = item.get("user", {}).get("username")
            if username:
                usernames.add(username)
    time.sleep(REQUEST_SLEEP)

    # Also pull commenters — they're likely raters
    data = trakt_get(f"/shows/{slug}/comments", params={"limit": limit})
    if data:
        for item in data:
            username = item.get("user", {}).get("username")
            if username:
                usernames.add(username)
    time.sleep(REQUEST_SLEEP)

    return list(usernames)[:limit]


def fetch_user_ratings(username: str) -> dict[str, float]:
    """
    Return {slug: rating} for all shows rated by this user.
    Uses GET /users/{username}/ratings/shows.
    """
    data = trakt_get(f"/users/{username}/ratings/shows")
    if not data:
        return {}
    result = {}
    for item in data:
        show = item.get("show", {})
        ids = show.get("ids", {})
        slug = ids.get("slug")
        rating = item.get("rating")
        if slug and rating is not None:
            result[slug] = float(rating)
    return result


def fetch_show_metadata(slug: str) -> dict:
    """
    Fetch extended metadata for a show.
    Returns dict with: slug, title, year, genres, status, runtime,
    aired_episodes, overview, trakt_rating, trakt_votes.
    """
    data = trakt_get(f"/shows/{slug}", params={"extended": "full"})
    if not data:
        return {"slug": slug}
    ids = data.get("ids", {})
    return {
        "slug": slug,
        "trakt_id": ids.get("trakt"),
        "imdb_id": ids.get("imdb"),
        "tmdb_id": ids.get("tmdb"),
        "title": data.get("title"),
        "year": data.get("year"),
        "genres": data.get("genres") or [],
        "status": data.get("status"),
        "runtime": data.get("runtime"),
        "aired_episodes": data.get("aired_episodes"),
        "overview": data.get("overview", ""),
        "trakt_rating": data.get("rating"),
        "trakt_votes": data.get("votes"),
    }


def run_ingest():
    if USER_RATINGS_OUT.exists() and SHOWS_META_OUT.exists():
        print("Ingested data already exists. Delete data/ files to re-fetch.")
        return

    # ── Step 1: Collect candidate usernames from seed shows ─────────────────
    if not USER_RATINGS_OUT.exists():
        print(f"\nStep 1: Collecting usernames from {len(SEED_SLUGS)} seed shows...")
        all_usernames: set[str] = set()
        for slug in tqdm(SEED_SLUGS, desc="Seed shows"):
            users = fetch_show_raters(slug)
            all_usernames.update(users)
            print(f"  {slug}: {len(users)} users  (total unique: {len(all_usernames)})")

        print(f"\nFound {len(all_usernames)} unique candidate users")

        # ── Step 2: Fetch their ratings ──────────────────────────────────────
        print(f"\nStep 2: Fetching ratings for up to {MAX_USERS} users...")
        user_rating_rows = []
        users_kept = 0

        for username in tqdm(list(all_usernames), desc="User ratings"):
            ratings = fetch_user_ratings(username)
            time.sleep(REQUEST_SLEEP)

            # Count overlap with seed slugs
            overlap = sum(1 for s in SEED_SLUGS if s in ratings)
            if overlap < MIN_SEED_OVERLAP:
                continue

            for slug, rating in ratings.items():
                user_rating_rows.append({
                    "username": username,
                    "slug": slug,
                    "rating": rating,
                })
            users_kept += 1
            if users_kept >= MAX_USERS:
                break

        print(f"\nKept {users_kept} users with >= {MIN_SEED_OVERLAP} seed overlap")
        print(f"Total rating rows: {len(user_rating_rows):,}")

        user_ratings_df = pd.DataFrame(user_rating_rows)
        user_ratings_df.to_parquet(USER_RATINGS_OUT, index=False)
        print(f"Saved to {USER_RATINGS_OUT.name}")
    else:
        print(f"User ratings already cached — loading.")
        user_ratings_df = pd.read_parquet(USER_RATINGS_OUT)

    # ── Step 3: Fetch show metadata ──────────────────────────────────────────
    if not SHOWS_META_OUT.exists():
        all_slugs = user_ratings_df["slug"].unique().tolist()
        print(f"\nStep 3: Fetching metadata for {len(all_slugs):,} shows...")

        meta_rows = []
        for slug in tqdm(all_slugs, desc="Show metadata"):
            meta = fetch_show_metadata(slug)
            meta_rows.append(meta)
            time.sleep(REQUEST_SLEEP)

        shows_meta_df = pd.DataFrame(meta_rows)
        shows_meta_df.to_parquet(SHOWS_META_OUT, index=False)
        print(f"Saved metadata for {len(shows_meta_df):,} shows to {SHOWS_META_OUT.name}")
    else:
        print("Show metadata already cached.")

    print("\nIngest complete.")


if __name__ == "__main__":
    run_ingest()
