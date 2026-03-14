# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

Two completely independent sub-projects. No shared code or data between them.

```
recs/
├── films/          ← MovieLens collaborative filtering for movies
│   ├── src/
│   ├── data/       ← parquet files + ml-25m/ dataset
│   ├── my_ratings.csv
│   └── .env        ← OMDB_API_KEY
│
├── shows/          ← Trakt-based collaborative filtering for TV shows
│   ├── src/
│   ├── data/       ← parquet files (user_ratings, shows_meta, my_neighbours, candidates)
│   └── .env        ← TRAKT_CLIENT_ID
│
└── split_ratings.py   ← one-time script to split IMDb export into films + TV
```

## Commands

```bash
# One-time: split IMDb export into films and TV
python split_ratings.py

# Films system (run from films/)
pip install -r requirements.txt
python src/ingest.py     # download + preprocess MovieLens 25M (~5–15 min, run once)
python src/match.py      # find 200 nearest neighbours (~2–10 min, run once)
python src/recommend.py
python src/recommend.py --no-interactive   # skip questions
python src/recommend.py --debug            # full scored candidate list

# Shows system (run from shows/)
pip install -r requirements.txt
python src/ingest.py     # fetch Trakt user ratings + show metadata (run once, needs TRAKT_CLIENT_ID)
python src/match.py      # find 100 nearest neighbours (run once or after updating MY_TV_RATINGS)
python src/recommend.py
python src/recommend.py --no-interactive
python src/recommend.py --debug
```

## Films system architecture (`films/`)

Five-stage pipeline:

1. **`src/ingest.py`** — Downloads MovieLens 25M (~250 MB), filters to users/movies with ≥50 ratings, saves three parquet files: `processed_ratings`, `movies_meta`, `movie_tags`. Also computes `imdb_ratings.parquet` — uses OMDb API if `OMDB_API_KEY` is set in `.env`, otherwise scales MovieLens average ratings as a proxy.

2. **`src/match.py`** — Loads `my_ratings.csv` (IMDb export, 1–10 scale), bridges to MovieLens IDs via `links.csv` imdbId, computes Pearson correlation against every active MovieLens user (0.5–5.0 scale). Takes a 3× initial pool by Pearson, then reranks by `combined_score = 0.6 × correlation + 0.4 × genre_alignment`. Genre alignment measures how much a neighbour's highly-rated films skew toward prestige genres (Drama/Crime/History…) vs blockbuster genres (Action/Sci-Fi/Fantasy…). Saves top 200 neighbours. Also exports `get_era_weighted_neighbours()` for session-scoped era filtering.

3. **`src/candidates.py`** — Gets all films rated ≥4.0★ by the 200 neighbours that the user hasn't seen. Uses `combined_score` for weighting. Computes `diversity_score = 0.7 × weighted_score + 0.3 × avg_neighbour_rating × top_neighbour_similarity`. Merges `imdb_ratings.parquet` so each candidate carries `imdb_rating`, `imdb_votes`, and `runtime`. Applies session filters (era, famousness, media type, runtime). `MIN_RECOMMENDER_COUNT` drops from 5 → 3 when an era filter is active. Contains the `ALREADY_SEEN` set (~280 normalised titles).

4. **`src/scorer.py`** — Deterministic taste scorer. **All tunable weights are named constants at the top of the file.** Six components: genre score (capped ±20/15), franchise penalty (−40 hard veto), IMDb gap signal via `imdb_gap_signal()` (±12/8/−10), positive keyword tags (cap +15), negative keyword tags (cap −20), era bonus (±2–5), plus a session mood modifier (±3–9). Classifies into `safe_bet` (≥45), `wild_card` (≥28), or `skip`.

5. **`src/recommend.py`** — Entry point. Interactive 5-question session via `get_session_preferences()`. Calls `get_era_weighted_neighbours()` when era filter is set (session-only, not saved). `--no-interactive` skips questions. `--debug` shows all 80 candidates. Saves JSON to `data/recommendations_YYYY-MM-DD.json`.

6. **`src/session.py`** — `SessionPrefs` dataclass. Imported by both `recommend.py` and `candidates.py` to avoid circular imports.

## Shows system architecture (`shows/`)

Same five-stage pipeline, adapted for Trakt TV data:

1. **`src/ingest.py`** — Seeds from `SEED_SLUGS` (15 hardcoded Trakt show slugs). Collects candidate users via Trakt `/shows/{slug}/lists` and `/shows/{slug}/comments` endpoints. Fetches show ratings for up to `MAX_USERS=300` users who share at least `MIN_SEED_OVERLAP=4` seed shows. Fetches extended metadata for all rated shows. Saves `user_ratings.parquet` and `shows_meta.parquet`. Both steps are idempotent.

2. **`src/match.py`** — **`MY_TV_RATINGS` dict (Trakt slug → 1–10) is hardcoded here** — edit this dict to add/update ratings, then re-run `match.py`. Also defines `TV_ALREADY_SEEN` set. Computes Pearson correlation with `MIN_OVERLAP=5`, reranks by genre alignment using same `0.6 × correlation + 0.4 × genre_alignment` formula. Saves top `K_NEIGHBOURS=100` neighbours.

3. **`src/candidates.py`** — Gets shows rated ≥8.0 (Trakt 1–10 scale) by neighbours. `MIN_RECOMMENDER_COUNT=3`, `MIN_AVG_RATING=7.0`. Applies session filters: era (year), status (ended/ongoing), episode count, and famousness by `trakt_votes`. Top 60 candidates by diversity score.

4. **`src/scorer.py`** — Same deterministic structure as films scorer. Signals: `TV_GENRE_SCORES` (Trakt genre strings, capped at 18), `TV_GENRE_COMBINATIONS` combos, Trakt community rating signal (≥8.5 → +6, <6.5 → −8), status signal (ended/cancelled → +4), neighbour consensus signal, episode count signal (6–60 eps sweet spot → +3, >150 → −3), keyword signals, mood modifier. Classifies into `safe_bet` (≥38), `wild_card` (≥24).

5. **`src/recommend.py`** — Same entry point pattern. 5-question interactive session: era, status (finished vs ongoing), episode commitment (≤15 / ≤50 / any), obscurity, mood. Saves JSON to `data/tv_recommendations_YYYY-MM-DD.json`.

6. **`src/session.py`** — `SessionPrefs` dataclass with `era_min_year`, `status_filter`, `max_episodes`, `exclude_famous`, `famous_votes_threshold`, `mood`.

## Key data details

### Films
- `links.csv` imdbId values may have leading zeros stripped — `match.py` normalises both sides with `lstrip("0")` before merging.
- MovieLens titles include year suffix (`"Toy Story (1995)"`) and article-last format (`"Dark Knight, The"`). The `norm()` function in `candidates.py` handles both.
- `imdb_ratings.parquet` stores `imdb_rating`, `imdb_votes`, and `runtime`. If using the proxy path (no OMDb key), `runtime` is `None` and `imdb_votes` is the MovieLens rating count.
- `SAFE_BET_FALLBACK_THRESHOLD` (38) must stay below `SAFE_BET_THRESHOLD` (45).
- If `data/imdb_ratings.parquet` was generated before `imdb_votes`/`runtime` columns were added, delete it and re-run `src/ingest.py`.

### Shows
- TV ratings are not loaded from a CSV — they live in the `MY_TV_RATINGS` dict in `shows/src/match.py`. To add new ratings: edit that dict and re-run `python src/match.py`.
- Trakt slugs are used as primary keys throughout (not numeric IDs).
- `TV_ALREADY_SEEN` in `match.py` is automatically seeded from `MY_TV_RATINGS` keys, plus any additional slugs to exclude.

## Tuning the scorer

### Films (`films/src/scorer.py`)
- `GENRE_SCORES` — per-genre base points
- `COMBO_BONUSES` — list of `(frozenset, bonus, reason_string)` tuples
- `imdb_gap_signal()` — standalone function, edit thresholds directly
- `POSITIVE_KEYWORDS` / `NEGATIVE_KEYWORDS` — tag-matching signals
- `SAFE_BET_THRESHOLD` / `WILD_CARD_THRESHOLD` — classification cutoffs (currently 45/28)

Run `python src/recommend.py --debug` after any weight change to verify Drama+Crime films score higher than Action+Sci-Fi films.

### Shows (`shows/src/scorer.py`)
- `TV_GENRE_SCORES` — per-genre base points (Trakt genre strings, lowercase)
- `TV_GENRE_COMBINATIONS` — list of `(frozenset, bonus, reason_string)` tuples
- `TRAKT_RATING_HIGH` / `TRAKT_RATING_LOW` — community rating thresholds
- `POSITIVE_KEYWORDS` / `NEGATIVE_KEYWORDS` — overview keyword signals
- `SAFE_BET_THRESHOLD` / `WILD_CARD_THRESHOLD` — classification cutoffs (currently 38/24)

Run `python src/recommend.py --debug` after any weight change to verify Drama+Crime shows score higher than Reality/Game-show content.
