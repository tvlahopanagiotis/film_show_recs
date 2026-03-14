"""
Deterministic taste scorer for TV show candidates.

All weights are named constants — tune here, not scattered in logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
SAFE_BET_THRESHOLD = 38
SAFE_BET_FALLBACK_THRESHOLD = 30   # must be < SAFE_BET_THRESHOLD
WILD_CARD_THRESHOLD = 24

# ── Genre weights (Trakt genre strings, lowercase) ────────────────────────────
TV_GENRE_SCORES: dict[str, int] = {
    "drama": 6,
    "crime": 7,
    "thriller": 6,
    "mystery": 6,
    "history": 5,
    "war": 5,
    "biography": 5,
    "documentary": 4,
    "mini-series": 3,
    "western": 3,
    "sport": 2,
    "comedy": 1,
    "science-fiction": 0,
    "action": 0,
    "adventure": -1,
    "family": -3,
    "animation": -3,
    "reality": -8,
    "game-show": -10,
    "talk-show": -10,
}

# Hard cap on the raw genre component so no single genre dominates the total score
GENRE_SCORE_CAP = 18

# Bonus for high-value genre pairs (frozenset → (bonus, reason))
TV_GENRE_COMBINATIONS: list[tuple[frozenset, int, str]] = [
    (frozenset({"drama", "crime"}),     6, "Crime drama"),
    (frozenset({"drama", "history"}),   5, "Historical drama"),
    (frozenset({"drama", "thriller"}),  4, "Thriller drama"),
    (frozenset({"drama", "mystery"}),   4, "Mystery drama"),
    (frozenset({"drama", "war"}),       4, "War drama"),
    (frozenset({"crime", "thriller"}),  3, "Crime thriller"),
    (frozenset({"drama", "biography"}), 3, "Biographical drama"),
]

# ── Trakt community rating ────────────────────────────────────────────────────
TRAKT_RATING_HIGH = 8.5   # score above this → bonus
TRAKT_RATING_HIGH_BONUS = 6
TRAKT_RATING_LOW = 6.5    # score below this → penalty
TRAKT_RATING_LOW_PENALTY = -8

# ── Status signal ─────────────────────────────────────────────────────────────
STATUS_ENDED_BONUS = 4      # complete story — can commit without risk of cancellation
STATUS_ONGOING_BONUS = 1

# ── Neighbour consensus signal ───────────────────────────────────────────────
# High avg neighbour rating = the people most like you loved it
NEIGHBOUR_RATING_HIGH = 8.5        # avg neighbour rating above this → bonus
NEIGHBOUR_RATING_HIGH_BONUS = 7
NEIGHBOUR_RATING_DECENT = 8.0      # above this → smaller bonus
NEIGHBOUR_RATING_DECENT_BONUS = 3

# ── Episode count signal ──────────────────────────────────────────────────────
# Prefer limited/compact series; penalise very long-runners
EPISODES_SWEET_SPOT_MIN = 6
EPISODES_SWEET_SPOT_MAX = 60
EPISODES_SWEET_SPOT_BONUS = 3
EPISODES_LONG_THRESHOLD = 150
EPISODES_LONG_PENALTY = -3

# ── Keyword signals ───────────────────────────────────────────────────────────
POSITIVE_KEYWORDS = [
    ("based on true", 3, "Based on true events"),
    ("political", 2, "Political themes"),
    ("psychological", 3, "Psychological depth"),
    ("critically acclaimed", 2, "Critically acclaimed"),
    ("limited series", 3, "Limited series"),
    ("anthology", 2, "Anthology format"),
    ("historical", 3, "Historical setting"),
    ("period", 2, "Period setting"),
    ("grief", 3, "Explores grief"),
    ("loss", 3, "Themes of loss"),
    ("redemption", 3, "Redemption arc"),
    ("morality", 4, "Moral complexity"),
    ("trauma", 3, "Explores trauma"),
    ("identity", 3, "Identity themes"),
    ("class", 3, "Class dynamics"),
    ("privilege", 3, "Privilege examined"),
    ("systemic", 3, "Systemic critique"),
    ("examination", 3, "In-depth examination"),
]

NEGATIVE_KEYWORDS = [
    ("sitcom", -3, "Sitcom format"),
    ("prank", -5, "Prank show"),
    ("contest", -3, "Contest format"),
    ("dating", -5, "Dating show"),
    ("celebrity", -3, "Celebrity focus"),
    ("marvel", -15, "Marvel franchise"),
    ("dc comics", -15, "DC franchise"),
    ("superhero", -12, "Superhero genre"),
]

# Franchise title markers applied directly against the show title
FRANCHISE_TITLE_MARKERS = ["marvel's", "dc's"]

# ── Mood modifiers ────────────────────────────────────────────────────────────
MOOD_BONUSES: dict[str, dict[str, int]] = {
    "tense": {
        "crime": 5,
        "thriller": 6,
        "mystery": 4,
        "drama": 2,
    },
    "thoughtful": {
        "drama": 5,
        "history": 5,
        "biography": 4,
        "documentary": 4,
        "mystery": 3,
    },
    "lighter": {
        "comedy": 8,
        "animation": 4,
        "adventure": 3,
        "family": 2,
        "drama": -3,
        "crime": -2,
        "thriller": -2,
    },
}


def score_show(
    title: str,
    genres: list,
    trakt_rating: Optional[float],
    status: Optional[str],
    aired_episodes: Optional[int],
    overview: str,
    avg_neighbour_rating: float,
    recommender_count: int,
    n_neighbours: int,
    top_neighbour_similarity: float,
    mood: str = "any",
) -> tuple[float, list[str], str]:
    """
    Score a candidate show. Returns (taste_score, reasons, category).

    category is "safe_bet" or "wild_card".
    """
    score = 0.0
    reasons = []

    # 0. Franchise title penalty (applied before genre scoring)
    title_lower = (title or "").lower()
    for marker in FRANCHISE_TITLE_MARKERS:
        if marker in title_lower:
            score -= 15
            reasons.append(f"Franchise title marker ({-15})")
            break

    # 1. Genre score (capped to prevent one genre from dominating)
    genre_set = set(g.lower() for g in (genres or []))
    genre_score = min(sum(TV_GENRE_SCORES.get(g, 0) for g in genre_set), GENRE_SCORE_CAP)
    if genre_score != 0:
        score += genre_score
        top_genre = max(genre_set, key=lambda g: TV_GENRE_SCORES.get(g, 0), default="")
        if top_genre:
            reasons.append(f"Genre: {top_genre.title()} ({genre_score:+d})")

    # 2. Genre combination bonuses
    for genre_combo, bonus, label in TV_GENRE_COMBINATIONS:
        if genre_combo.issubset(genre_set):
            score += bonus
            reasons.append(f"{label} combination (+{bonus})")

    # 3. Trakt community rating signal
    if trakt_rating is not None:
        if trakt_rating >= TRAKT_RATING_HIGH:
            score += TRAKT_RATING_HIGH_BONUS
            reasons.append(f"Trakt community rating {trakt_rating:.1f} (+{TRAKT_RATING_HIGH_BONUS})")
        elif trakt_rating < TRAKT_RATING_LOW:
            score += TRAKT_RATING_LOW_PENALTY
            reasons.append(f"Low Trakt rating {trakt_rating:.1f} ({TRAKT_RATING_LOW_PENALTY})")

    # 4. Status signal
    if status in ("ended", "canceled"):
        score += STATUS_ENDED_BONUS
        reasons.append(f"Complete series ({STATUS_ENDED_BONUS:+d})")
    elif status in ("returning series", "in production"):
        score += STATUS_ONGOING_BONUS

    # 5. Neighbour consensus signal
    if avg_neighbour_rating >= NEIGHBOUR_RATING_HIGH:
        score += NEIGHBOUR_RATING_HIGH_BONUS
        reasons.append(f"Neighbours rated it {avg_neighbour_rating:.1f}★ avg (+{NEIGHBOUR_RATING_HIGH_BONUS})")
    elif avg_neighbour_rating >= NEIGHBOUR_RATING_DECENT:
        score += NEIGHBOUR_RATING_DECENT_BONUS
        reasons.append(f"Neighbours rated it {avg_neighbour_rating:.1f}★ avg (+{NEIGHBOUR_RATING_DECENT_BONUS})")

    # 7. Episode count signal
    if aired_episodes is not None:
        if EPISODES_SWEET_SPOT_MIN <= aired_episodes <= EPISODES_SWEET_SPOT_MAX:
            score += EPISODES_SWEET_SPOT_BONUS
            reasons.append(f"{aired_episodes} episodes — compact commitment (+{EPISODES_SWEET_SPOT_BONUS})")
        elif aired_episodes > EPISODES_LONG_THRESHOLD:
            score += EPISODES_LONG_PENALTY
            reasons.append(f"{aired_episodes} episodes — long commitment ({EPISODES_LONG_PENALTY})")

    # 8. Overview keyword signals
    overview_lower = (overview or "").lower()
    for kw, bonus, label in POSITIVE_KEYWORDS:
        if kw in overview_lower:
            score += bonus
            reasons.append(f"{label} (+{bonus})")
    for kw, penalty, label in NEGATIVE_KEYWORDS:
        if kw in overview_lower:
            score += penalty
            reasons.append(f"{label} ({penalty})")

    # 9. Mood modifier
    if mood != "any" and mood in MOOD_BONUSES:
        mood_delta = sum(MOOD_BONUSES[mood].get(g, 0) for g in genre_set)
        if mood_delta != 0:
            score += mood_delta
            reasons.append(f"Mood match ({mood_delta:+d})")

    # Clamp
    score = max(0.0, min(100.0, score))

    if score >= SAFE_BET_THRESHOLD:
        category = "safe_bet"
    elif score >= WILD_CARD_THRESHOLD:
        category = "wild_card"
    else:
        category = "below_threshold"

    return score, reasons, category


def score_and_select(candidates: "pd.DataFrame", mood: str = "any") -> dict:
    """
    Score all candidates and split into safe_bets and wild_cards.
    Returns {"safe_bets": [...], "wild_cards": [...], "all_scored": [...]}.
    """
    n_neighbours = len(candidates)  # rough proxy; actual count passed separately
    scored = []

    for _, row in candidates.iterrows():
        genres_raw = row.get("genres")
        if genres_raw is None or (hasattr(genres_raw, '__len__') and len(genres_raw) == 0):
            genres = []
        elif isinstance(genres_raw, (list, np.ndarray)):
            genres = list(genres_raw)
        else:
            genres = []

        taste_score, reasons, category = score_show(
            title=row.get("title", ""),
            genres=genres,
            trakt_rating=row.get("trakt_rating"),
            status=row.get("status"),
            aired_episodes=row.get("aired_episodes"),
            overview=row.get("overview", ""),
            avg_neighbour_rating=row.get("avg_neighbour_rating", 0),
            recommender_count=row.get("recommender_count", 0),
            n_neighbours=n_neighbours,
            top_neighbour_similarity=row.get("top_neighbour_similarity", 0),
            mood=mood,
        )

        scored.append({
            "slug": row.get("slug", ""),
            "title": row.get("title", ""),
            "year": row.get("year"),
            "genres": genres,
            "status": row.get("status"),
            "aired_episodes": row.get("aired_episodes"),
            "runtime": row.get("runtime"),
            "trakt_rating": row.get("trakt_rating"),
            "trakt_votes": row.get("trakt_votes"),
            "taste_score": taste_score,
            "category": category,
            "reasons": reasons,
            "recommended_by": row.get("recommender_count", 0),
            "neighbour_score": row.get("avg_neighbour_rating", 0),
        })

    scored.sort(key=lambda x: x["taste_score"], reverse=True)

    safe_bets = [s for s in scored if s["category"] == "safe_bet"]
    wild_cards = [s for s in scored if s["category"] == "wild_card"]

    # Fallback: if no safe bets, lower the bar
    if not safe_bets:
        fallback = [s for s in scored if s["taste_score"] >= SAFE_BET_FALLBACK_THRESHOLD]
        safe_bets = fallback[:3]

    return {
        "safe_bets": safe_bets[:3],
        "wild_cards": wild_cards[:2],
        "all_scored": scored,
    }
