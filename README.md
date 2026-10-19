# recs

A personal recommendation system for films and TV shows, built on collaborative filtering. Two completely independent pipelines — one powered by the MovieLens ml-32m dataset (covers up to 2023), one by Trakt.tv — each with a CLI, a Streamlit dashboard, and a Stremio addon.

---

## What it does

Both systems work the same way:

1. Find users in a large dataset who have similar taste to yours (your "neighbours")
2. Collect everything they rated highly that you haven't seen
3. Score each candidate using a deterministic, tunable taste scorer
4. Output **Safe Bets** (high confidence matches), **Wild Cards** (lower confidence but potentially interesting), and **Recent Picks** (2020+ films discovered via TMDB recommendations)

The scoring is entirely local and deterministic — no LLMs, no external recommendation APIs. Every weight and threshold is a named constant you can tune.

---

## Project layout

```
recs/
├── films/              ← MovieLens ml-32m collaborative filtering for movies
│   ├── src/
│   │   ├── ingest.py       # download + preprocess MovieLens dataset
│   │   ├── match.py        # find your 400 nearest neighbours
│   │   ├── candidates.py   # collect films your neighbours loved
│   │   ├── scorer.py       # deterministic taste scorer (all weights here)
│   │   ├── recommend.py    # interactive CLI entry point
│   │   ├── tmdb_recent.py  # fetch post-2020 candidates via TMDB recommendations
│   │   └── session.py      # SessionPrefs dataclass
│   ├── data/               # parquet files + ml-32m/ dataset (generated)
│   ├── my_ratings.csv      # your IMDb export goes here
│   └── .env                # TMDB_API_KEY (required for post-2020 picks), OMDB_API_KEY (optional)
│
├── shows/              ← Trakt-based collaborative filtering for TV shows
│   ├── src/
│   │   ├── ingest.py       # fetch Trakt user ratings + show metadata
│   │   ├── match.py        # load TV ratings from films/my_ratings.csv + find your 200 nearest neighbours
│   │   ├── candidates.py   # collect shows your neighbours loved
│   │   ├── scorer.py       # deterministic taste scorer (all weights here)
│   │   ├── recommend.py    # interactive CLI entry point
│   │   └── session.py      # SessionPrefs dataclass
│   ├── data/               # parquet files (generated)
│   └── .env                # TRAKT_CLIENT_ID
│
├── app/                ← Streamlit dashboard module
│   ├── pipeline.py         # wires films/shows pipelines for the dashboard
│   ├── data_loader.py      # cached parquet/CSV I/O
│   ├── charts.py           # Plotly figure factories
│   ├── components.py       # Streamlit UI blocks (cards, badges)
│   └── parse_reasons.py    # score reason string parser
│
├── .streamlit/
│   └── config.toml         # binds to 0.0.0.0 for LAN access
│
├── app.py              ← Streamlit dashboard entry point
├── addon.py            ← Stremio addon server
├── split_ratings.py    ← one-time utility to split IMDb export
├── requirements.txt    ← core pipeline dependencies
└── requirements_app.txt ← dashboard dependencies
```

---

## Setup

### 1. Install dependencies

```bash
# Core pipeline (needed for CLI usage)
pip install -r requirements.txt

# Dashboard + Stremio addon (adds streamlit and plotly)
pip install -r requirements_app.txt
```

### 2. Films system setup

**Export your IMDb ratings:**
Go to your IMDb profile → Lists → Your Ratings → Export. Place the CSV at `films/my_ratings.csv`.

**Create `films/.env`:**
```
TMDB_API_KEY=your_key_here      # required for post-2020 picks and runtime/vote data
OMDB_API_KEY=your_key_here      # optional — TMDB is used instead when present
```
Free TMDB key: themoviedb.org/settings/api. Free OMDb key: omdbapi.com (1,000 req/day, not needed when TMDB is set).

**Run the pipeline once (~20–40 min total):**
```bash
cd films
python src/ingest.py       # download + preprocess ml-32m (~5–15 min), then TMDB enrichment (~35 min)
python src/match.py        # find your 400 nearest neighbours (~1 min)
python src/tmdb_recent.py  # fetch post-2020 candidates from TMDB (~2 min)
```

All three steps are idempotent — re-running skips work already done. After setup, recommendations run in seconds.

**Refresh post-2020 candidates** (run monthly or after updating your ratings):
```bash
cd films && python src/tmdb_recent.py --force
```

### 3. Shows system setup

**Get a Trakt API key:**
Create an app at trakt.tv/oauth/applications (free). Copy the Client ID into `shows/.env`:
```
TRAKT_CLIENT_ID=your_client_id_here
```

**Add your TV ratings:**
Nothing extra to do: the shows pipeline reads the TV Series / TV Mini Series rows of the same IMDb export as films (`films/my_ratings.csv`) and matches them to Trakt by IMDb ID. `match.py` prints any shows it couldn't match (shows that aren't in `shows_meta.parquet`).

**Run the pipeline once (~2–5 min):**
```bash
cd shows
python src/ingest.py    # fetch Trakt data (1–2 h; resumes if interrupted)
python src/match.py     # find your 200 nearest neighbours
```

---

## Usage

### CLI (films)

```bash
cd films
python src/recommend.py                  # interactive questions then recommendations
python src/recommend.py --no-interactive # run with default settings (no filters)
python src/recommend.py --debug          # show all candidates with full scores
```

The interactive session asks about era, famousness preference, mood, and runtime. Answers apply session-only filters — nothing is saved to disk, so each run is independent.

Results are saved to `films/data/recommendations_YYYY-MM-DD.json`.

### CLI (shows)

```bash
cd shows
python src/recommend.py                  # 5 interactive questions
python src/recommend.py --no-interactive
python src/recommend.py --debug          # show all 60 candidates
```

The interactive session asks about era, show status (finished vs. ongoing), episode commitment, famousness preference, and mood.

Results are saved to `shows/data/tv_recommendations_YYYY-MM-DD.json`.

### Streamlit dashboard

```bash
# from recs/
streamlit run app.py
```

The dashboard exposes both pipelines in a single UI:

- **Sidebar** — switch between Films/Shows mode, set filters, click "Get Recommendations"
- **Tab 1 — Tonight's Picks** — Safe Bets, Wild Cards, and Recent Picks (2020+) as cards with score breakdown
- **Tab 2 — All Candidates** — Interactive scatter plot (diversity vs. taste score). Click any point to inspect it. Sortable table below.
- **Tab 3 — Taste Profile** — Your average rating per genre for both films and shows

On first load the dashboard reads the latest saved JSON so it's never blank.

**Access from your phone (same WiFi):**
```bash
ipconfig getifaddr en0   # find your Mac's local IP, e.g. 192.168.1.42
streamlit run app.py
# on phone: http://192.168.1.42:8501
```
The `.streamlit/config.toml` already binds to `0.0.0.0`, so no extra flags needed.

### Static GitHub Pages site

For phone access away from your computer, export a static public snapshot:

```bash
# from recs/, after films/shows ingest + match are complete
python3 scripts/export_site.py
python3 -m http.server 8000 -d docs   # optional local preview
```

The exporter runs the fast recommendation pipelines and writes `docs/data/recs.json`.
The static site in `docs/` shows Tonight's Picks, All Candidates, and Taste Profile
without exposing `.env` files, raw IMDb exports, parquet files, or MovieLens/Trakt data.

To publish with GitHub Pages, commit and push `docs/`, then set Pages source to
`main` / `/docs` in the repository settings. Updates are manual: rerun the exporter,
commit the changed `docs/data/recs.json`, and push.

### Stremio addon

Serves your recommendations as browsable Stremio catalogs with poster images.

```bash
# from recs/
python addon.py            # starts on port 7000
python addon.py --port 7777
```

**Install in Stremio:**
Addons → search icon → "Add addon from URL" → `http://localhost:7000/manifest.json`

**From phone (same WiFi):**
```bash
ipconfig getifaddr en0   # find your Mac's IP
# install from: http://192.168.1.42:7000/manifest.json
```

The addon runs both pipelines at startup with default settings and caches results in memory. Both "My Film Picks" and "My Show Picks" catalogs appear in Stremio's Discover view with poster images. Restart the addon to refresh recommendations.

---

## How the scoring works

### Finding neighbours

Both systems compute Pearson correlation between your ratings and every user in the dataset, then **shrink it for users with little overlap**:

```
similarity = correlation × min(overlap, N) / N        # N = 50 films, 20 shows
```

Without this, users who happen to agree with you on a handful of titles outrank genuine matches, and the problem grows with the number of users. Films keeps 400 neighbours; shows keeps 200 and still reranks by a genre-alignment score (prestige vs blockbuster genres).

### Ranking candidates

**Films.** Every film at least 2 neighbours have rated (any score) is a candidate. Its ranking (`diversity_score`) is a 50/50 blend of two percentiles:

- **quality** — how far above their own average *all* MovieLens users rate it
- **personal prediction** — the same, among your neighbours only, weighted by similarity and shrunk when few neighbours rated it

Ratings are compared to each rater's own average, so harsh and generous raters count equally. Titles with no TMDB *movie* data are dropped — they're TV series that MovieLens lists as films. Top 80 go to the scorer, plus post-2020 films from the TMDB recent cache.

**Shows.** Shows rated ≥ 8.0 by at least 3 neighbours; `diversity_score` = 0.7 × similarity-weighted rating sum + 0.3 × average rating × similarity of the closest endorsers. Top 60 go to the scorer.

### Post-2020 films (films only)

MovieLens (ml-32m) runs to 2023. For anything newer, `tmdb_recent.py` builds a separate pool by calling TMDB's recommendations endpoint for your top-rated films, filtering to 2020+, and scoring the results with the taste scorer.

### Choosing the picks

`python scripts/evaluate.py` hides part of your ratings and checks whether each ranking puts the hidden titles you loved above the ones you disliked. The candidate ranking beats the hand-tuned taste scorer clearly (films ~0.89 vs ~0.74 AUC, shows ~0.78 vs ~0.68), so:

- **Safe bets** (and shows' recent picks) are the top of the candidate ranking, minus vetoes. Films veto franchise/superhero titles, animation and documentary-only titles; shows veto anything the taste scorer rates below 15.
- **Wild cards** are the taste scorer's best picks among the less-famous half of the pool — the discovery slot.
- **Recent picks** (films) are the taste scorer's best 2020+ titles.

One caveat: the evaluation can only use titles you chose to watch, so it rewards generally acclaimed titles and can't measure discovery — that's what the vetoes and wild cards are for.

### The taste scorer

The scorer applies a stack of signals to produce a `taste_score`, used for wild cards, film recent picks and vetoes. All weights are named constants at the top of `films/src/scorer.py` and `shows/src/scorer.py`.

**Films scorer signals:**

| Signal | Range | Notes |
|---|---|---|
| Genre score | ±20 | Per-genre points, capped |
| Genre combinations | varies | Drama+Crime, Animation+Drama, Sci-Fi+Drama, etc. |
| Neighbour consensus | up to +7 | Avg neighbour rating ≥4.5★ → +7, ≥4.2★ → +3 |
| IMDb gap signal | ±12 | Penalises overrated spectacle; rewards underseen prestige gems |
| Franchise penalty | −40 | Marvel/DC/superhero keywords (whole-word match on title + tags) |
| Sequel penalty | −10 | "Part 2"-style title markers, unless genre is Drama/Crime |
| Positive keywords | up to +15 | "psychological", "dark", "based on true story", etc. |
| Negative keywords | up to −20 | "slapstick", "blockbuster", "feel-good", etc. |
| Era bonus | ±2–5 | Boosts classics (1950–75: +5, 1976–99: +3) and recent (2020+: +2) |
| Mood modifier | ±3–9 | Genre bonuses when mood filter is set |

**Shows scorer signals:**

| Signal | Range | Notes |
|---|---|---|
| Genre score | ±18 | Per-genre points, capped |
| Genre combinations | varies | Crime+Drama, Drama+History, etc. |
| Community rating | ±8 | Trakt ≥8.5 → +6, <6.5 → −8 |
| Neighbour consensus | up to +7 | Avg neighbour rating ≥8.5 → +7, ≥8.0 → +3 |
| Show status | ±4 | Ended/cancelled → +4 (complete story) |
| Episode count | ±3 | 6–60 eps sweet spot → +3, >150 → −3 |
| Era bonus | +2 | 2020+ shows get a small boost |
| Positive keywords | up to +6 | "psychological", "based on true story", etc. |
| Negative keywords | up to −15 | "sitcom", "marvel", etc. |
| Mood modifier | ±5–8 | Applied when mood filter is set |

**Badge thresholds** (taste score, for titles not picked): films safe ≥ 30 / wild ≥ 15; shows safe ≥ 38 / wild ≥ 24.

---

## Updating your ratings

**Films:** Replace `films/my_ratings.csv` with a fresh IMDb export, then re-run:
```bash
cd films
python src/match.py
python src/tmdb_recent.py --force   # refresh post-2020 pool with new seed films
```

**Shows:** Uses the same `films/my_ratings.csv`, so after replacing it, re-run:
```bash
cd shows && python src/match.py
```

Neighbours recompute in a few minutes; `ingest.py` does not need to re-run for either system.

---

## Tuning the scorer

Edit the weight constants at the top of `scorer.py` in either pipeline, then verify with debug mode:

```bash
cd films && python src/recommend.py --debug
cd shows && python src/recommend.py --debug
```

Check that Drama+Crime titles score higher than Action+Sci-Fi titles. Adjust constants until the ordering looks right.

---

## One-time utilities

**Split a mixed IMDb export into films and TV:**
```bash
python split_ratings.py
```
Outputs `films/my_ratings.csv` and a TV ratings file.

---

## Data citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 1–19.

If the automatic MovieLens download fails, get `ml-32m.zip` from grouplens.org/datasets/movielens/, extract so that `films/data/ml-32m/ratings.csv` exists, then run `ingest.py`.
