"""
Generate and score candidate films from nearest neighbours.
"""

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_RATINGS = DATA_DIR / "processed_ratings.parquet"
MOVIES_META = DATA_DIR / "movies_meta.parquet"
MOVIE_TAGS = DATA_DIR / "movie_tags.parquet"
MY_NEIGHBOURS = DATA_DIR / "my_neighbours.parquet"
IMDB_RATINGS = DATA_DIR / "imdb_ratings.parquet"
TMDB_ENRICHED = DATA_DIR / "tmdb_enriched.parquet"
CANDIDATES_OUT = DATA_DIR / "candidates.parquet"

# Minimum number of neighbours who must have rated a film 4.0+ for it to qualify
MIN_RECOMMENDER_COUNT = 5
MIN_AVG_NEIGHBOUR_RATING = 3.8
NEIGHBOUR_LIKE_THRESHOLD = 4.0  # MovieLens 0.5–5.0 scale
TOP_N_CANDIDATES = 80


# ── Already-seen filter ───────────────────────────────────────────────────────

def norm(title: str) -> str:
    """Normalise a title for fuzzy deduplication."""
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"^(.*),\s*(the|a|an)\s*$", r"\2 \1", title.lower())
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


_ALREADY_SEEN_RAW = [
    "The Martian", "Casino Royale", "Guardians of the Galaxy Vol 2",
    "Spider-Man Homecoming", "Captain America Civil War", "Deadpool", "Ray",
    "Kill Bill Vol 1", "Leon The Professional", "Doctor Strange",
    "Star Wars Return of the Jedi", "Lord of the Rings Two Towers",
    "Star Wars Empire Strikes Back", "Birdman", "Selma", "Big Fish", "Fury",
    "American Beauty", "Django Unchained", "The Bear", "Mr Robot",
    "The Mandalorian", "The Grand Tour", "Black Mirror", "Ted Lasso",
    "Succession", "Clarksons Farm", "Better Call Saul", "Rick and Morty",
    "The Sopranos", "The Wire", "True Detective", "Stranger Things", "Lincoln",
    "Black Hawk Down", "The Best Offer", "Spider-Man Far from Home", "Lupin",
    "You", "Dont Look Up", "No Time to Die", "Avatar", "Back to the Future",
    "Rogue One", "Dune Part One", "Love Actually", "Mollys Game",
    "The Accountant", "Squid Game", "In Time", "El Camino", "Mindhunter",
    "The Queens Gambit", "Vice", "Sherlock Holmes A Game of Shadows",
    "Notting Hill", "The Notebook", "Dumb and Dumber", "About Time",
    "Bruce Almighty", "Ted 2", "Horrible Bosses", "Midnight in Paris",
    "The Hangover Part II", "Ted", "The Town", "Sherlock Holmes",
    "Two and a Half Men", "The 40 Year Old Virgin", "The Nice Guys",
    "Parks and Recreation", "War Dogs", "The Great Gatsby", "Troy",
    "Oceans Thirteen", "Oceans Twelve", "Oceans Eleven", "The Hangover",
    "Suits", "BoJack Horseman", "The Silence of the Lambs", "The Wave",
    "Gravity", "Limitless", "The Shawshank Redemption", "Furious 7",
    "The Revenant", "12 Angry Men", "Batman Begins", "Angels and Demons",
    "Fantastic Beasts and Where to Find Them", "Assassins Creed", "Logan",
    "The Circle", "Wonder Woman", "Star Wars The Last Jedi", "Now You See Me",
    "Knives Out", "Money Heist", "Once Upon a Time in Hollywood", "The Witcher",
    "The Gentlemen", "Apollo 13", "1917", "The Meg", "The Last Dance",
    "Quantum of Solace", "Men in Black", "The Big Bang Theory", "Glass",
    "Ant-Man", "Die Hard 2", "Kill Bill Vol 2", "The Post", "Klaus",
    "Moneyball", "The Artist", "Sweeney Todd", "Edge of Tomorrow",
    "Now You See Me 2", "Kung Fu Panda 2", "The Terminal", "Death Note",
    "Westworld", "Rome", "Narcos", "Friends", "The World at War",
    "Brooklyn Nine-Nine", "The Lives of Others", "Hot Fuzz", "Mad Men",
    "Cast Away", "Shaun of the Dead", "Jojo Rabbit", "Parasite", "Killing Eve",
    "Chernobyl", "Breaking Bad", "Game of Thrones", "The Two Popes",
    "The Irishman", "Avengers Endgame", "Joker", "Ford v Ferrari", "WALL-E",
    "Kung Fu Panda", "Monsters Inc", "The Incredibles", "Ratatouille",
    "Unbreakable", "Split", "Inglourious Basterds", "Peaky Blinders",
    "How to Train Your Dragon 2", "How to Train Your Dragon", "House of Cards",
    "Deadpool 2", "Tinker Tailor Soldier Spy", "Darkest Hour", "The Hateful Eight",
    "Thor Ragnarok", "The Cabin in the Woods", "Finding Nemo", "Good Will Hunting",
    "Alien", "Ex Machina", "It", "Dunkirk", "Baby Driver", "Frozen", "Casino",
    "Glengarry Glen Ross", "The Perks of Being a Wallflower", "Independence Day",
    "Magnolia", "Judgment at Nuremberg", "The Truman Show", "Into the Wild",
    "Get Out", "Contact", "Dead Poets Society", "Good Morning Vietnam",
    "Dr Strangelove", "The Last Samurai", "The Social Network", "Up in the Air",
    "The Shining", "Indiana Jones and the Last Crusade", "Skyfall", "The Green Mile",
    "Die Hard with a Vengeance", "Catch Me If You Can", "Black Swan", "Snatch",
    "Amelie", "American Psycho", "Your Name", "There Will Be Blood", "Arrival",
    "Schindlers List", "Downfall", "Sherlock", "Her", "Little Miss Sunshine",
    "Eyes Wide Shut", "Tropic Thunder", "The Da Vinci Code", "Inside Out",
    "V for Vendetta", "A Beautiful Mind", "Star Wars The Force Awakens", "Heat",
    "Die Hard", "LA Confidential", "Reservoir Dogs", "Braveheart", "The Prestige",
    "The Intouchables", "Raiders of the Lost Ark", "Star Wars A New Hope",
    "Goodfellas", "Spotlight", "Zootopia", "Trumbo", "Bridge of Spies",
    "The Big Short", "High Fidelity", "Memento", "Whiplash", "The Pianist",
    "The Departed", "American History X", "The Usual Suspects", "Silicon Valley",
    "The Dark Knight Rises", "Kingsman The Secret Service", "The Avengers",
    "Guardians of the Galaxy", "Avengers Age of Ultron", "Mad Max Fury Road",
    "Argo", "Spectre", "Persepolis", "Se7en", "Lord of the Rings Fellowship of the Ring",
    "The Dark Knight", "Saving Private Ryan", "Gladiator",
    "Lord of the Rings Return of the King", "Inception", "The Matrix",
    "Pulp Fiction", "Fight Club", "Forrest Gump", "Wild Tales", "Anchorman",
    "Donnie Darko", "The Grand Budapest Hotel", "Interstellar", "The Imitation Game",
    "Boyhood", "The Wolf of Wall Street", "500 Days of Summer", "Full Metal Jacket",
    "Coach Carter", "The Kings Speech",
    # Classics
    "Taxi Driver", "Fargo", "Trainspotting", "The Godfather", "The Godfather Part II",
    "One Flew Over the Cuckoos Nest", "Amadeus", "Citizen Kane", "Casablanca", "Carlitos Way",
    # Post-2010 likely already seen
    "Shutter Island", "Gone Girl", "Drive", "12 Years a Slave", "Nightcrawler", "Prisoners",
]

ALREADY_SEEN: set = {norm(t) for t in _ALREADY_SEEN_RAW}


def _top_20_mean(sims: list) -> float:
    """Mean similarity of the top-20% endorsers — rewards films championed by the closest neighbours."""
    if not sims:
        return 0.0
    sorted_sims = sorted(sims, reverse=True)
    top_n = max(1, len(sorted_sims) // 5)
    return sum(sorted_sims[:top_n]) / top_n


def generate_candidates(my_ratings: dict, prefs=None, neighbours=None) -> pd.DataFrame:
    """
    Find films loved by nearest neighbours that the user hasn't seen.
    Applies session filters from prefs (era, famousness, media type, runtime).
    Returns a ranked DataFrame enriched with metadata and tags.
    """
    # Import here to avoid circular import at module level
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from session import SessionPrefs
    if prefs is None:
        prefs = SessionPrefs()

    t0 = time.time()

    if not MY_NEIGHBOURS.exists():
        raise FileNotFoundError(
            f"{MY_NEIGHBOURS} not found. Run python src/match.py first."
        )

    if neighbours is None:
        neighbours = pd.read_parquet(MY_NEIGHBOURS)
    print(f"Loaded {len(neighbours)} neighbours")

    # Era-filtered runs have a smaller eligible pool per neighbour — lower the bar
    min_recommender_count = 3 if prefs.era_min_year is not None else MIN_RECOMMENDER_COUNT

    ratings = pd.read_parquet(PROCESSED_RATINGS)
    movies_meta = pd.read_parquet(MOVIES_META)

    # Merge in tags
    if MOVIE_TAGS.exists():
        movie_tags = pd.read_parquet(MOVIE_TAGS)
        movies_meta = movies_meta.merge(movie_tags, on="movieId", how="left")
        movies_meta["tags_str"] = movies_meta["tags_str"].fillna("")
    else:
        movies_meta["tags_str"] = ""

    neighbour_ids = set(neighbours["userId"].tolist())

    # Use combined_score (post genre-alignment) if available, else fall back to correlation
    score_col = "combined_score" if "combined_score" in neighbours.columns else "correlation"
    corr_map = dict(zip(neighbours["userId"], neighbours[score_col]))

    # Get all ratings by my neighbours
    neighbour_ratings = ratings[ratings["userId"].isin(neighbour_ids)].copy()

    # Keep only films neighbours liked (≥ 4.0★) that I haven't rated
    my_movie_ids = set(my_ratings.keys())
    liked = neighbour_ratings[
        (neighbour_ratings["rating"] >= NEIGHBOUR_LIKE_THRESHOLD)
        & (~neighbour_ratings["movieId"].isin(my_movie_ids))
    ].copy()

    liked["similarity"] = liked["userId"].map(corr_map)
    liked["weighted_score"] = liked["rating"] * liked["similarity"]

    # Collect per-movie similarity lists for the diversity boost
    movie_sim_lists = liked.groupby("movieId")["similarity"].apply(list)

    # Aggregate across neighbours
    agg = liked.groupby("movieId").agg(
        weighted_score=("weighted_score", "sum"),
        recommender_count=("userId", "nunique"),
        avg_neighbour_rating=("rating", "mean"),
    ).reset_index()

    agg["coverage"] = agg["recommender_count"] / len(neighbours)

    # Diversity boost: favour films championed by the most similar neighbours
    agg["top_neighbour_similarity"] = agg["movieId"].map(
        movie_sim_lists.apply(_top_20_mean)
    )

    # Apply base filters
    agg = agg[
        (agg["recommender_count"] >= min_recommender_count)
        & (agg["avg_neighbour_rating"] >= MIN_AVG_NEIGHBOUR_RATING)
    ]

    # Enrich with metadata
    agg = agg.merge(movies_meta, on="movieId", how="left")

    # diversity_score: 70% collaborative breadth + 30% quality-of-endorsement
    agg["diversity_score"] = (
        agg["weighted_score"] * 0.7
        + agg["avg_neighbour_rating"] * agg["top_neighbour_similarity"] * 0.3
    )

    # Merge IMDb proxy ratings (ML-average based fallback)
    if IMDB_RATINGS.exists():
        imdb_ratings = pd.read_parquet(IMDB_RATINGS)
        imdb_ratings["movieId"] = imdb_ratings["movieId"].astype(agg["movieId"].dtype)
        agg = agg.merge(imdb_ratings, on="movieId", how="left")
    else:
        agg["imdb_rating"] = None

    # Merge TMDB enrichment (real ratings, runtime, vote counts) — prefers TMDB over proxy
    if TMDB_ENRICHED.exists():
        tmdb = pd.read_parquet(TMDB_ENRICHED)
        tmdb["movieId"] = tmdb["movieId"].astype(agg["movieId"].dtype)
        tmdb = tmdb.rename(columns={"imdb_rating": "imdb_rating_tmdb"})
        agg = agg.merge(tmdb, on="movieId", how="left")
        # Real TMDB vote_average wins over proxy where available
        agg["imdb_rating"] = agg["imdb_rating_tmdb"].fillna(agg["imdb_rating"])
        agg.drop(columns=["imdb_rating_tmdb"], inplace=True, errors="ignore")

    # Unified fame signal: tmdb_vote_count preferred, imdb_votes proxy as fallback
    if "tmdb_vote_count" in agg.columns:
        agg["fame_signal"] = agg["tmdb_vote_count"]
        if "imdb_votes" in agg.columns:
            agg["fame_signal"] = agg["fame_signal"].fillna(agg["imdb_votes"])
    elif "imdb_votes" in agg.columns:
        agg["fame_signal"] = agg["imdb_votes"]
    else:
        agg["fame_signal"] = None

    coverage = agg["imdb_rating"].notna().sum()
    print(f"IMDb rating coverage: {coverage} / {len(agg)} candidates")

    # Normalise title for already-seen check
    agg["title_norm"] = agg["title"].fillna("").apply(norm)
    agg = agg[~agg["title_norm"].isin(ALREADY_SEEN)]

    # ── Session filters ───────────────────────────────────────────────────────

    # Era filter
    if prefs.era_min_year is not None:
        agg = agg[agg["year"].isna() | (agg["year"] >= prefs.era_min_year)]
    if prefs.era_max_year is not None:
        agg = agg[agg["year"].isna() | (agg["year"] <= prefs.era_max_year)]

    # Famousness filter — uses fame_signal (tmdb_vote_count preferred, imdb_votes fallback)
    if prefs.exclude_famous and "fame_signal" in agg.columns:
        agg = agg[
            agg["fame_signal"].isna()
            | (agg["fame_signal"] < prefs.famous_votes_threshold)
        ]
    elif prefs.exclude_famous:
        print("  Note: fame signal unavailable — famousness filter skipped. Re-run ingest.py.")

    # Runtime filter — populated by TMDB enrichment in ingest.py
    if prefs.max_runtime is not None and "runtime" in agg.columns:
        agg = agg[
            agg["runtime"].isna()
            | (agg["runtime"] <= prefs.max_runtime)
        ]

    # Sort by diversity_score and take top N
    agg = agg.sort_values("diversity_score", ascending=False).head(TOP_N_CANDIDATES)
    agg = agg.reset_index(drop=True)

    agg.to_parquet(CANDIDATES_OUT, index=False)
    print(f"Generated {len(agg)} candidates in {time.time()-t0:.1f}s")

    if len(agg) > 0:
        print("\nTop 20 candidates by diversity score:")
        cols = [
            "title", "recommender_count", "avg_neighbour_rating",
            "top_neighbour_similarity", "weighted_score", "diversity_score",
        ]
        available = [c for c in cols if c in agg.columns]
        print(agg.head(20)[available].to_string(index=False))

    return agg
