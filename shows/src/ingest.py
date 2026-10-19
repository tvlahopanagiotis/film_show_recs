"""
Ingest TV show ratings from Trakt.tv via public API.

Strategy:
  1. Seed from my top-rated shows — find users who listed or commented on them.
  2. Collect all their show ratings; keep users who rated enough of my shows.
  3. Fetch metadata for every show that could become a candidate.

Run once: python src/ingest.py
Rebuild into a separate folder: python src/ingest.py --out-dir data_new --max-users 3000
Requires TRAKT_CLIENT_ID in .env (free at trakt.tv/oauth/applications).

Progress is checkpointed in <out-dir>/ingest_checkpoint/, so an interrupted run
resumes where it stopped. Metadata already in data/shows_meta.parquet is reused.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

SHOWS_DIR = Path(__file__).parent.parent
DATA_DIR = SHOWS_DIR / "data"

TRAKT_API_KEY = os.getenv("TRAKT_CLIENT_ID", "").strip()
TRAKT_BASE = "https://api.trakt.tv"

# Fallback seeds, used when my ratings can't be loaded yet (no shows_meta on a first run)
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
    "the-queen-s-gambit",
    "narcos",
    "westworld",
    "money-heist",
    "squid-game",
    "house-of-cards",
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

SEED_MIN_RATING = 7     # my shows rated at least this become seeds
MIN_MY_OVERLAP = 5      # user must have rated at least this many of my shows (= match.MIN_OVERLAP)
MAX_USERS = 3000        # stop once this many users are kept
PAGES_PER_SEED = 4      # pages of lists and of comments to scan per seed show
PAGE_LIMIT = 250        # Trakt caps list pages at 250 items
META_MIN_RATERS = 3     # only fetch metadata for shows that could pass MIN_RECOMMENDER_COUNT
REQUEST_SLEEP = 0.3     # seconds between API requests (well within rate limit)
MAX_ATTEMPTS = 8        # retries per request on rate limits / server errors before giving up


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
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.get(
                f"{TRAKT_BASE}{path}",
                headers=headers,
                params=params or {},
                timeout=20,
            )
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 10))
            print(f"  Rate limited — sleeping {retry_after}s", flush=True)
            time.sleep(retry_after + 1)
            continue
        if r.status_code == 403:
            raise RuntimeError("Trakt returned 403 — TRAKT_CLIENT_ID is invalid or the app was revoked.")
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if not r.ok:
            return None
        return r.json()
    # Don't let a temporary outage look like "no data" — that would be checkpointed
    raise RuntimeError(f"Trakt still failing after {MAX_ATTEMPTS} attempts on {path}. Re-run to resume.")


def load_seeds() -> tuple[list[str], set[str]]:
    """Return (seed slugs, all my rated slugs) — from my ratings if possible, else SEED_SLUGS."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from match import load_my_ratings
        my_ratings = load_my_ratings()
    except FileNotFoundError:
        print("Can't load my ratings yet — falling back to SEED_SLUGS")
        return SEED_SLUGS, set(SEED_SLUGS)
    seeds = sorted(s for s, r in my_ratings.items() if r >= SEED_MIN_RATING)
    return seeds, set(my_ratings)


def fetch_show_raters(slug: str, pages: int) -> set[str]:
    """
    Return usernames of people engaged with this show.

    Trakt doesn't expose a raters-list endpoint, so use the owners of public
    lists containing the show, plus its commenters — both are likely raters.
    """
    usernames = set()
    for endpoint in (f"/shows/{slug}/lists/personal/added", f"/shows/{slug}/comments"):
        for page in range(1, pages + 1):
            data = trakt_get(endpoint, params={"limit": PAGE_LIMIT, "page": page})
            time.sleep(REQUEST_SLEEP)
            if not data:
                break
            for item in data:
                username = item.get("user", {}).get("username")
                if username:
                    usernames.add(username)
            if len(data) < PAGE_LIMIT:
                break
    return usernames


def fetch_user_ratings(username: str) -> dict[str, float]:
    """
    Return {slug: rating} for all shows rated by this user.
    Uses GET /users/{username}/ratings/shows (unpaginated — returns everything).
    """
    data = trakt_get(f"/users/{username}/ratings/shows")
    if not isinstance(data, list):
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
    if not isinstance(data, dict):
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


def run_ingest(out_dir: Path, max_users: int, pages: int):
    user_ratings_out = out_dir / "user_ratings.parquet"
    shows_meta_out = out_dir / "shows_meta.parquet"
    if user_ratings_out.exists() and shows_meta_out.exists():
        print(f"Ingested data already exists in {out_dir}. Delete it (or use --out-dir) to re-fetch.")
        return

    ckpt = out_dir / "ingest_checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    candidates_file = ckpt / "candidate_users.json"
    raw_file = ckpt / "user_ratings.jsonl"
    meta_file = ckpt / "shows_meta.jsonl"

    seeds, my_slugs = load_seeds()

    # ── Step 1: Collect candidate usernames from seed shows ─────────────────
    if candidates_file.exists():
        candidates = json.loads(candidates_file.read_text())
        print(f"Step 1: {len(candidates):,} candidate users loaded from checkpoint")
    else:
        print(f"\nStep 1: Collecting usernames from {len(seeds)} seed shows...")
        seen_with = Counter()
        for slug in tqdm(seeds, desc="Seed shows"):
            users = fetch_show_raters(slug, pages)
            seen_with.update(users)
            tqdm.write(f"  {slug}: {len(users)} users  (total unique: {len(seen_with)})")
        # Users found via several seeds are the likeliest to overlap with me — fetch them first
        candidates = [u for u, _ in seen_with.most_common()]
        candidates_file.write_text(json.dumps(candidates))
        print(f"Found {len(candidates):,} unique candidate users")

    # ── Step 2: Fetch their ratings (checkpointed per user) ─────────────────
    done: dict[str, dict] = {}
    if raw_file.exists():
        for line in raw_file.read_text().splitlines():
            rec = json.loads(line)
            done[rec["username"]] = rec["ratings"]

    def overlap(ratings: dict) -> int:
        return sum(1 for s in ratings if s in my_slugs)

    kept = sum(1 for r in done.values() if overlap(r) >= MIN_MY_OVERLAP)
    todo = [u for u in candidates if u not in done]
    print(f"\nStep 2: Fetching ratings (keep users with >= {MIN_MY_OVERLAP} of my {len(my_slugs)} shows)")
    print(f"  {len(done):,} already fetched, {kept:,} kept so far; target {max_users:,}")

    with open(raw_file, "a") as f, tqdm(todo, desc="User ratings") as bar:
        for username in bar:
            if kept >= max_users:
                break
            ratings = fetch_user_ratings(username)
            time.sleep(REQUEST_SLEEP)
            f.write(json.dumps({"username": username, "ratings": ratings}) + "\n")
            f.flush()
            done[username] = ratings
            if overlap(ratings) >= MIN_MY_OVERLAP:
                kept += 1
                bar.set_postfix(kept=kept)

    rows = [
        {"username": u, "slug": s, "rating": r}
        for u, ratings in done.items() if overlap(ratings) >= MIN_MY_OVERLAP
        for s, r in ratings.items()
    ]
    user_ratings_df = pd.DataFrame(rows)
    print(f"\nKept {user_ratings_df['username'].nunique():,} users, {len(user_ratings_df):,} rating rows")

    # ── Step 3: Fetch show metadata ──────────────────────────────────────────
    raters = user_ratings_df.groupby("slug")["username"].nunique()
    needed = set(raters[raters >= META_MIN_RATERS].index) | my_slugs

    known = []
    if (DATA_DIR / "shows_meta.parquet").exists():
        known.append(pd.read_parquet(DATA_DIR / "shows_meta.parquet"))
    if meta_file.exists():
        known.append(pd.DataFrame([json.loads(l) for l in meta_file.read_text().splitlines()]))
    have = set(pd.concat(known)["slug"]) if known else set()
    missing = sorted(needed - have)

    print(f"\nStep 3: Metadata for {len(needed):,} shows rated by >= {META_MIN_RATERS} users "
          f"({len(needed) - len(missing):,} cached, fetching {len(missing):,})...")
    with open(meta_file, "a") as f:
        for slug in tqdm(missing, desc="Show metadata"):
            f.write(json.dumps(fetch_show_metadata(slug)) + "\n")
            f.flush()
            time.sleep(REQUEST_SLEEP)

    known.append(pd.DataFrame([json.loads(l) for l in meta_file.read_text().splitlines()]))
    shows_meta_df = pd.concat(known).drop_duplicates("slug", keep="last")
    shows_meta_df = shows_meta_df[shows_meta_df["slug"].isin(needed)].reset_index(drop=True)

    user_ratings_df.to_parquet(user_ratings_out, index=False)
    shows_meta_df.to_parquet(shows_meta_out, index=False)
    print(f"Saved {user_ratings_out} and {shows_meta_out} ({len(shows_meta_df):,} shows)")
    print("\nIngest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Trakt user ratings + show metadata")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                        help="output folder (relative to shows/); default data/")
    parser.add_argument("--max-users", type=int, default=MAX_USERS)
    parser.add_argument("--pages", type=int, default=PAGES_PER_SEED,
                        help="pages of lists/comments to scan per seed show")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else SHOWS_DIR / args.out_dir
    run_ingest(out_dir, args.max_users, args.pages)
