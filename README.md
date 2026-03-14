# recs

A personal recommendation system for films and TV shows, built on collaborative filtering. Two completely independent pipelines — one powered by the MovieLens 25M dataset, one by Trakt.tv — each with a CLI, a Streamlit dashboard, and a Stremio addon.

---

## What it does

Both systems work the same way:

1. Find users in a large dataset who have similar taste to yours (your "neighbours")
2. Collect everything they rated highly that you haven't seen
3. Score each candidate using a deterministic, tunable taste scorer
4. Output **Safe Bets** (high confidence matches) and **Wild Cards** (lower confidence but potentially interesting)

The scoring is entirely local and deterministic — no LLMs, no external recommendation APIs. Every weight and threshold is a named constant you can tune.

---

## Project layout

```
recs/
├── films/              ← MovieLens 25M collaborative filtering for movies
│   ├── src/
│   │   ├── ingest.py       # download + preprocess MovieLens dataset
│   │   ├── match.py        # find your 200 nearest neighbours
│   │   ├── candidates.py   # collect films your neighbours loved
│   │   ├── scorer.py       # deterministic taste scorer (all weights here)
│   │   ├── recommend.py    # interactive CLI entry point
│   │   └── session.py      # SessionPrefs dataclass
│   ├── data/               # parquet files + ml-25m/ dataset (generated)
│   ├── my_ratings.csv      # your IMDb export goes here
│   └── .env                # OMDB_API_KEY (optional)
│
├── shows/              ← Trakt-based collaborative filtering for TV shows
│   ├── src/
│   │   ├── ingest.py       # fetch Trakt user ratings + show metadata
│   │   ├── match.py        # find your 100 nearest neighbours + MY_TV_RATINGS dict
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

**Optional — real IMDb ratings via OMDb:**
Create `films/.env`:
```
OMDB_API_KEY=your_key_here
```
Free keys at omdbapi.com (1,000 req/day). Without a key, the system uses a MovieLens proxy — works fine.

**Run the pipeline once (~15–25 min total):**
```bash
cd films
python src/ingest.py    # download + preprocess MovieLens 25M (~5–15 min)
python src/match.py     # find your 200 nearest neighbours (~2–10 min)
```

Both steps are idempotent — re-running skips work already done. After setup, recommendations run in seconds.

### 3. Shows system setup

**Get a Trakt API key:**
Create an app at trakt.tv/oauth/applications (free). Copy the Client ID into `shows/.env`:
```
TRAKT_CLIENT_ID=your_client_id_here
```

**Add your TV ratings:**
Open `shows/src/match.py` and edit the `MY_TV_RATINGS` dict at the top of the file. Keys are Trakt slugs (the URL segment from `trakt.tv/shows/breaking-bad`), values are your rating 1–10:
```python
MY_TV_RATINGS = {
    "breaking-bad": 10,
    "the-wire": 10,
    "true-detective": 9,
    # add your own...
}
```

**Run the pipeline once (~2–5 min):**
```bash
cd shows
python src/ingest.py    # fetch Trakt data
python src/match.py     # find your 100 nearest neighbours
```

---

## Usage

### CLI (films)

```bash
cd films
python src/recommend.py                  # 4 interactive questions then recommendations
python src/recommend.py --no-interactive # run with default settings (no filters)
python src/recommend.py --debug          # show all 80 candidates with full scores
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
- **Tab 1 — Tonight's Picks** — Safe Bets and Wild Cards as cards with score breakdown
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

Both systems compute Pearson correlation between your ratings and every user in the dataset, requiring a minimum overlap of rated items. After getting the top candidates by correlation, they are reranked by a combined score:

```
combined_score = 0.6 × correlation + 0.4 × genre_alignment
```

Genre alignment measures how much a neighbour's highly-rated content skews toward prestige genres (Drama, Crime, History, Documentary) vs. blockbuster genres (Action, Sci-Fi, Fantasy). This surfaces neighbours who don't just correlate with your ratings but share your underlying taste axis.

Films keeps 200 neighbours; shows keeps 100.

### Generating candidates

Films rated ≥ 4.0★ (out of 5) by neighbours, or shows rated ≥ 8.0 (out of 10), that you haven't seen. Each candidate gets a diversity score:

```
diversity_score = 0.7 × weighted_neighbour_score + 0.3 × avg_rating × top_neighbour_similarity
```

The 0.3 component rewards content championed by your *closest* neighbours, not just broadly popular ones. Top 80 films / 60 shows are passed to the scorer.

### Scoring

The scorer applies a stack of signals to produce a final `taste_score`. All weights are named constants at the top of `films/src/scorer.py` and `shows/src/scorer.py` — edit them and re-run `--debug` to verify the effect.

**Films scorer signals:**

| Signal | Range | Notes |
|---|---|---|
| Genre score | ±20 | Per-genre points, capped |
| Genre combinations | varies | Drama+Crime, Animation+Drama, etc. |
| IMDb gap signal | ±12 | Detects overrated spectacle / hidden prestige gems |
| Franchise penalty | −40 | Hard veto for Marvel/DC/superhero titles |
| Sequel penalty | −10 | Applied unless genre is Drama/Crime |
| Positive keywords | up to +15 | "psychological", "dark", "based on true story", etc. |
| Negative keywords | up to −20 | "slapstick", "blockbuster", "feel-good", etc. |
| Era bonus | ±2–5 | Boosts classics or recent depending on session era filter |
| Mood modifier | ±3–9 | Genre bonuses when mood filter is set |

**Shows scorer signals:**

| Signal | Range | Notes |
|---|---|---|
| Genre score | ±18 | Per-genre points, capped |
| Genre combinations | varies | Crime+Drama, Drama+History, etc. |
| Community rating | ±8 | Trakt ≥8.5 → +6, <6.5 → −8 |
| Neighbour consensus | up to +7 | Avg neighbour rating ≥8.5 → +7 |
| Show status | ±4 | Ended/cancelled → +4 (complete story) |
| Episode count | ±3 | 6–60 eps sweet spot → +3, >150 → −3 |
| Positive keywords | up to +6 | "psychological", "based on true story", etc. |
| Negative keywords | up to −15 | "sitcom", "marvel", etc. |
| Mood modifier | ±5–8 | Applied when mood filter is set |

**Classification thresholds:**

| | Safe Bet | Wild Card |
|---|---|---|
| Films | ≥ 45 | ≥ 28 |
| Shows | ≥ 38 | ≥ 24 |

If fewer than 2 safe bets are found, the fallback threshold drops to 38 (films) / 30 (shows).

---

## Updating your ratings

**Films:** Replace `films/my_ratings.csv` with a fresh IMDb export, then re-run:
```bash
cd films && python src/match.py
```

**Shows:** Edit `MY_TV_RATINGS` in `shows/src/match.py`, then re-run:
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

If the automatic MovieLens download fails, get `ml-25m.zip` from grouplens.org/datasets/movielens/25m/, extract so that `films/data/ml-25m/ratings.csv` exists, then run `ingest.py`.
