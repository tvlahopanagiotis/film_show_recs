"""
Deterministic taste scoring engine.
All weights are named constants at the top — tune them here.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ── Genre scores ──────────────────────────────────────────────────────────────
# Based on average viewer ratings per genre from personal IMDb data

GENRE_SCORES: dict[str, int] = {
    "History":       9,
    "Biography":     8,
    "Drama":         7,
    "Crime":         6,
    "War":           6,
    "Mystery":       5,
    "Music":         4,
    "Thriller":      3,
    "Sport":         3,
    "Documentary":   3,
    "Romance":       0,
    "Western":       0,
    "Adventure":     0,
    "Comedy":       -2,
    "Fantasy":      -3,
    "Horror":       -4,
    "Action":       -5,
    "Sci-Fi":       -6,
}
# Animation is deliberately absent from GENRE_SCORES — it is handled as a standalone
# post-cap penalty (see ANIMATION_PENALTY) so it cannot be absorbed by the genre cap
# when an animated film also carries Drama/History/War.

GENRE_SCORE_CAP_POSITIVE = 20
GENRE_SCORE_CAP_NEGATIVE = -15

# ── Animation penalty ─────────────────────────────────────────────────────────
# The handful of animated titles rated highly in my_ratings.csv are exceptions I
# made, not evidence of a preference. Left unpenalised, collaborative filtering
# reads those few high scores as "loves animation" and floods the recommendations.
# Applied AFTER the genre cap so a Drama/History-tagged animated film can't hide
# behind its other genres. The IMDb tiers are the "exceptional ones" escape hatch:
# a genuinely acclaimed animated film is docked far less, but never rewarded.
ANIMATION_PENALTY = -25                 # default: animation is a hard negative
ANIMATION_ACCLAIMED_IMDB = 8.0          # ≥ this → treated as a possible exception
ANIMATION_ACCLAIMED_PENALTY = -10
ANIMATION_EXCEPTIONAL_IMDB = 8.5        # ≥ this → the rare film worth an exception
ANIMATION_EXCEPTIONAL_PENALTY = -4
# Family + Animation is the kids-movie signal — the furthest thing from my taste
FAMILY_WITH_ANIMATION_PENALTY = -8

# Genre combination bonuses
COMBO_BONUSES: list[tuple[frozenset, int, str]] = [
    (frozenset({"Drama", "Crime"}),     7, "Drama + Crime combination (+7) — core taste sweet spot (Breaking Bad, Mindhunter)"),
    (frozenset({"Drama", "History"}),   5, "Drama + History combination (+5) — prestige history (Chernobyl, Schindler's List)"),
    (frozenset({"Comedy", "Crime"}),    3, "Comedy + Crime combination (+3) — smart genre blend (Knives Out, Hot Fuzz)"),
]
# Sci-Fi + Drama without Action
SCIFI_DRAMA_NO_ACTION_BONUS = 4
SCIFI_DRAMA_NO_ACTION_REASON = "Sci-Fi + Drama without Action (+4) — ideas-first sci-fi (Arrival, Ex Machina)"

# ── Franchise / blockbuster penalties ────────────────────────────────────────

FRANCHISE_PENALTY = -40
SEQUEL_PENALTY = -10

FRANCHISE_KEYWORDS: list[str] = [
    "marvel", "mcu", "avengers", "dc comics", "dceu", "superhero",
    "transformers", "fast and furious", "jurassic",
    "x-men", "spider-man", "batman v superman",
]
SEQUEL_KEYWORDS: list[str] = ["sequel", "part 2", "part ii", "part iii"]

# ── IMDb consensus gap ────────────────────────────────────────────────────────

IMDb_ACTION_HIGH_PENALTY = -12     # high IMDb on pure action/spectacle = red flag
IMDb_PRESTIGE_BONUS = 8            # moderately-rated prestige = potential hidden gem
IMDb_LOW_PENALTY = -10             # genuinely low-rated

# ── Keyword signals ───────────────────────────────────────────────────────────

POSITIVE_KEYWORDS: dict[str, int] = {
    "psychological":        5,
    "moral ambiguity":      5,
    "based on true story":  4,
    "based on true events": 4,
    "true crime":           4,
    "character study":      4,
    "dark":                 3,
    "tension":              3,
    "conspiracy":           3,
    "investigative":        3,
    "period piece":         3,
    "historical":           3,
    "political":            3,
    "satire":               3,
    "surreal":              2,
    "twist ending":         2,
    "nonlinear":            2,
    "dystopia":             2,
}
POSITIVE_KEYWORD_CAP = 15

NEGATIVE_KEYWORDS: dict[str, int] = {
    "slapstick":            -8,
    "gross-out":            -8,
    "teen comedy":          -8,
    "blockbuster":          -8,
    "style over substance": -8,
    "slasher":              -8,
    "romantic comedy":      -6,
    "comedy of errors":     -6,
    "jump scare":           -6,
    "gore":                 -6,
    "parody":               -5,
    "spoof":                -5,
    "road trip comedy":     -5,
    "action-packed":        -5,
    "crowd pleaser":        -5,
    "camp":                 -5,
    "buddy comedy":         -4,
    "torture":              -6,
    "visually stunning":    -3,
}
NEGATIVE_KEYWORD_CAP = -20

# ── Era bonus ─────────────────────────────────────────────────────────────────

ERA_BONUSES: list[tuple[tuple[int, int], int, str]] = [
    ((1950, 1975), 5, "Classic era 1950–1975 (+5) — Judgment at Nuremberg, Dr Strangelove"),
    ((1976, 1999), 3, "Golden era 1976–1999 (+3) — Goodfellas, Pulp Fiction, Heat"),
    ((2000, 2015), 2, "Contemporary prestige 2000–2015 (+2)"),
    ((2020, 2099), 2, "Recent era 2020+ (+2) — counteracts MovieLens data sparsity for new releases"),
]

# ── Neighbour consensus ───────────────────────────────────────────────────────
# score_film() has no access to collaborative data, so this is applied
# in score_candidates() after the base score is computed.
NEIGHBOUR_RATING_HIGH = 4.5          # avg neighbour rating, MovieLens 0.5–5.0 scale
NEIGHBOUR_RATING_HIGH_BONUS = 7
NEIGHBOUR_RATING_DECENT = 4.2
NEIGHBOUR_RATING_DECENT_BONUS = 3

# ── Recent picks ──────────────────────────────────────────────────────────────
RECENT_ERA_MIN_YEAR = 2020

# ── Classification thresholds ─────────────────────────────────────────────────

SAFE_BET_THRESHOLD = 30
WILD_CARD_THRESHOLD = 15
SAFE_BET_FALLBACK_THRESHOLD = 20  # fallback when fewer than 2 safe bets found (must be < SAFE_BET_THRESHOLD)


# ── Core scoring ──────────────────────────────────────────────────────────────

def _parse_genres(genres) -> list[str]:
    """Accept a list or pipe-separated string; return a clean list."""
    if isinstance(genres, list):
        return [g.strip() for g in genres if g.strip()]
    if not genres or (isinstance(genres, float)):
        return []
    return [g.strip() for g in str(genres).split("|") if g.strip() and g != "(no genres listed)"]


def _tags_lower(tags_str) -> str:
    if not tags_str or (isinstance(tags_str, float)):
        return ""
    return str(tags_str).lower()


def imdb_gap_signal(imdb_rating: Optional[float], genres: list[str]) -> tuple[int, Optional[str]]:
    """
    Compute the IMDb consensus gap component.

    The viewer's taste consistently diverges from IMDb consensus in a predictable direction:
    - Spectacle films (Action/Sci-Fi/Fantasy) with very high IMDb scores are overrated for this taste
    - Prestige films (Drama/Crime/History etc.) in the 7.0–8.2 range are often underseen gems
    - Genuinely low-rated films are a reliable quality signal downward

    Returns (score_delta, reason_string | None).
    """
    if imdb_rating is None:
        return 0, None
    try:
        rating = float(imdb_rating)
    except (TypeError, ValueError):
        return 0, None
    if pd.isna(rating):
        return 0, None

    genre_set = set(genres)

    # Spectacle bait: high IMDb on action/fantasy/sci-fi without dramatic anchor
    # Pattern: Revenant 8.0 → viewer 3; Mad Max 8.1 → viewer 2; Kill Bill 8.2 → viewer 5
    if (
        rating >= 8.0
        and genre_set & {"Action", "Sci-Fi", "Fantasy"}
        and not genre_set & {"Drama", "Crime"}
    ):
        return (
            IMDb_ACTION_HIGH_PENALTY,
            f"High IMDb ({rating:.1f}) on spectacle genre ({IMDb_ACTION_HIGH_PENALTY}) — Revenant/Mad Max pattern",
        )

    # Hidden gem prestige: moderately-rated films in prestige genres
    # Pattern: Apollo 13 7.7 → viewer 9; Darkest Hour 7.4 → viewer 9; Hot Fuzz 7.8 → viewer 9
    if (
        7.0 <= rating <= 8.2
        and genre_set & {"History", "Biography", "Drama", "Crime", "War", "Mystery"}
    ):
        return (
            IMDb_PRESTIGE_BONUS,
            f"IMDb {rating:.1f} in prestige genre (+{IMDb_PRESTIGE_BONUS}) — underseen gem pattern (Apollo 13, Darkest Hour)",
        )

    # Genuinely low-rated: real quality signal downward
    if rating < 6.5:
        return IMDb_LOW_PENALTY, f"Low IMDb rating {rating:.1f} ({IMDb_LOW_PENALTY})"

    return 0, None


def score_film(
    title: str,
    genres,
    tags_str,
    imdb_rating: Optional[float] = None,
    year=None,
    mood: str = "any",
) -> dict:
    """
    Score a film against the viewer's taste profile.

    Returns:
        {
            "taste_score": float (0–100),
            "category": "safe_bet" | "wild_card" | "skip",
            "reasons": list[str],
            "franchise_penalty": int,
        }
    """
    genres_list = _parse_genres(genres)
    genre_set = set(genres_list)
    tags = _tags_lower(tags_str)
    title_lower = (title or "").lower()
    combined_text = title_lower + " " + tags

    reasons: list[str] = []
    score = 0.0
    franchise_penalty = 0

    # ── Component 1: Genre score ───────────────────────────────────────────
    genre_score = 0
    for g in genre_set:
        if g == "Family":
            continue  # scored only via the animation component below
        genre_score += GENRE_SCORES.get(g, 0)

    genre_score = max(GENRE_SCORE_CAP_NEGATIVE, min(GENRE_SCORE_CAP_POSITIVE, genre_score))
    score += genre_score
    if genre_score > 0:
        top_genres = sorted(
            [(g, GENRE_SCORES.get(g, 0)) for g in genre_set if GENRE_SCORES.get(g, 0) > 0],
            key=lambda x: -x[1],
        )
        if top_genres:
            reasons.append(f"Genre profile (+{genre_score}): {', '.join(g for g, _ in top_genres[:3])}")
    elif genre_score < 0:
        bot_genres = sorted(
            [(g, GENRE_SCORES.get(g, 0)) for g in genre_set if GENRE_SCORES.get(g, 0) < 0],
            key=lambda x: x[1],
        )
        if bot_genres:
            reasons.append(f"Genre profile ({genre_score}): {', '.join(g for g, _ in bot_genres[:3])}")

    # Genre combination bonuses
    for combo, bonus, reason in COMBO_BONUSES:
        if combo.issubset(genre_set):
            score += bonus
            reasons.append(reason)

    if "Sci-Fi" in genre_set and "Drama" in genre_set and "Action" not in genre_set:
        score += SCIFI_DRAMA_NO_ACTION_BONUS
        reasons.append(SCIFI_DRAMA_NO_ACTION_REASON)

    # ── Component 1b: Animation penalty (applied after the genre cap) ──────
    if "Animation" in genre_set:
        if imdb_rating is not None and imdb_rating >= ANIMATION_EXCEPTIONAL_IMDB:
            animation_penalty = ANIMATION_EXCEPTIONAL_PENALTY
            animation_note = f"exceptional animation, IMDb {imdb_rating:.1f}"
        elif imdb_rating is not None and imdb_rating >= ANIMATION_ACCLAIMED_IMDB:
            animation_penalty = ANIMATION_ACCLAIMED_PENALTY
            animation_note = f"acclaimed animation, IMDb {imdb_rating:.1f}"
        else:
            animation_penalty = ANIMATION_PENALTY
            animation_note = "animation is not my thing"
        if "Family" in genre_set:
            animation_penalty += FAMILY_WITH_ANIMATION_PENALTY
            animation_note += " + family/kids"
        score += animation_penalty
        reasons.append(f"Animated ({animation_penalty}) — {animation_note}")

    # ── Component 2: Franchise / blockbuster penalty ───────────────────────
    matched_franchise = [kw for kw in FRANCHISE_KEYWORDS if kw in combined_text]
    if matched_franchise:
        franchise_penalty = FRANCHISE_PENALTY
        score += franchise_penalty
        reasons.append(f"Franchise/superhero markers detected ({franchise_penalty}): {matched_franchise[0]}")
    else:
        matched_sequel = [kw for kw in SEQUEL_KEYWORDS if kw in combined_text]
        if matched_sequel:
            drama_crime_score = sum(GENRE_SCORES.get(g, 0) for g in ["Drama", "Crime"] if g in genre_set)
            if drama_crime_score <= 10:
                franchise_penalty = SEQUEL_PENALTY
                score += franchise_penalty
                reasons.append(f"Sequel markers ({franchise_penalty}): {matched_sequel[0]}")

    # ── Component 3: IMDb consensus gap signal ─────────────────────────────
    imdb_score, imdb_reason = imdb_gap_signal(imdb_rating, genres_list)
    score += imdb_score
    if imdb_reason:
        reasons.append(imdb_reason)

    # ── Component 4: Positive keyword signals ─────────────────────────────
    pos_total = 0
    for kw, pts in POSITIVE_KEYWORDS.items():
        if kw in tags:
            pos_total += pts
            reasons.append(f"{kw.title()} tag (+{pts})")
    pos_total = min(POSITIVE_KEYWORD_CAP, pos_total)
    score += pos_total

    # ── Component 5: Negative keyword signals ─────────────────────────────
    neg_total = 0
    for kw, pts in NEGATIVE_KEYWORDS.items():
        if kw in tags:
            neg_total += pts
            reasons.append(f"'{kw}' tag ({pts})")
    neg_total = max(NEGATIVE_KEYWORD_CAP, neg_total)
    score += neg_total

    # ── Component 6: Era bonus ─────────────────────────────────────────────
    if year is not None:
        try:
            yr = int(year)
            for (lo, hi), bonus, reason in ERA_BONUSES:
                if lo <= yr <= hi:
                    score += bonus
                    reasons.append(reason)
                    break
        except (ValueError, TypeError):
            pass

    # Clamp
    taste_score = max(0.0, min(100.0, score))

    # ── Mood modifier ──────────────────────────────────────────────────────
    # Applied after the base score as a bonus/penalty on top — session-specific only.
    mood_bonus = 0
    if mood == "tense":
        tense_tags = {
            "tension", "thriller", "suspense", "psychological", "true crime",
            "conspiracy", "investigative", "twist ending",
        }
        if tags and any(t in tags for t in tense_tags):
            mood_bonus += 6
        if "Thriller" in genre_set or "Crime" in genre_set:
            mood_bonus += 3
        if mood_bonus:
            reasons.append(f"Mood match: tense/gripping (+{mood_bonus})")

    elif mood == "thoughtful":
        thoughtful_tags = {
            "moral ambiguity", "character study", "philosophical",
            "psychological", "political", "satire", "based on true story",
        }
        if tags and any(t in tags for t in thoughtful_tags):
            mood_bonus += 6
        if genre_set & {"History", "Biography", "Drama"}:
            mood_bonus += 3
        if mood_bonus:
            reasons.append(f"Mood match: thought-provoking (+{mood_bonus})")

    elif mood == "lighter":
        heavy_tags = {"dark", "torture", "gore", "bleak", "disturbing"}
        light_tags = {"comedy", "satire", "witty", "feel-good"}
        if tags and any(t in tags for t in heavy_tags):
            mood_bonus = -8
            reasons.append(f"Mood conflict: heavy content, lighter mood requested ({mood_bonus})")
        elif tags and any(t in tags for t in light_tags):
            mood_bonus = 5
            reasons.append(f"Mood match: lighter tone (+{mood_bonus})")

    taste_score = max(0.0, min(100.0, taste_score + mood_bonus))
    category = classify(taste_score, franchise_penalty)

    return {
        "taste_score": round(taste_score, 1),
        "category": category,
        "reasons": reasons,
        "franchise_penalty": franchise_penalty,
    }


def classify(taste_score: float, franchise_penalty: int) -> str:
    if franchise_penalty <= -35:
        return "skip"
    if taste_score >= SAFE_BET_THRESHOLD:
        return "safe_bet"
    if taste_score >= WILD_CARD_THRESHOLD:
        return "wild_card"
    return "skip"


def _year_int(y) -> int:
    try:
        return int(y)
    except (TypeError, ValueError):
        return 0


def score_candidates(candidates: pd.DataFrame, mood: str = "any") -> dict:
    """
    Score all candidates and return the final selection dict.

    Returns:
        {
            "safe_bets": [...],
            "wild_cards": [...],
            "recent_picks": [...],   # top 2020+ films regardless of overall bucket
            "all_scored": [...],
        }
    """
    scored_rows = []

    for _, row in candidates.iterrows():
        result = score_film(
            title=row.get("title", ""),
            genres=row.get("genres", ""),
            tags_str=row.get("tags_str", ""),
            imdb_rating=row.get("imdb_rating"),   # populated by candidates.py via imdb_ratings.parquet
            year=row.get("year"),
            mood=mood,
        )

        # Neighbour consensus bonus — score_film() is genre/tag/IMDb only, so the
        # collaborative signal is injected here where we have access to it.
        avg_rating = float(row.get("avg_neighbour_rating", 0))
        if avg_rating >= NEIGHBOUR_RATING_HIGH:
            result["taste_score"] = min(100.0, result["taste_score"] + NEIGHBOUR_RATING_HIGH_BONUS)
            result["reasons"].append(f"Neighbours rated it {avg_rating:.1f}★ avg (+{NEIGHBOUR_RATING_HIGH_BONUS})")
        elif avg_rating >= NEIGHBOUR_RATING_DECENT:
            result["taste_score"] = min(100.0, result["taste_score"] + NEIGHBOUR_RATING_DECENT_BONUS)
            result["reasons"].append(f"Neighbours rated it {avg_rating:.1f}★ avg (+{NEIGHBOUR_RATING_DECENT_BONUS})")

        result["category"] = classify(result["taste_score"], result["franchise_penalty"])

        scored_rows.append({
            "title": row.get("title", ""),
            "year": row.get("year"),
            "genres": _parse_genres(row.get("genres", "")),
            "imdbId": row.get("imdbId"),
            "taste_score": result["taste_score"],
            "category": result["category"],
            "reasons": result["reasons"],
            "franchise_penalty": result["franchise_penalty"],
            "neighbour_score": round(float(row.get("avg_neighbour_rating", 0)), 2),
            "recommended_by": int(row.get("recommender_count", 0)),
            "weighted_score": round(float(row.get("weighted_score", 0)), 2),
        })

    all_scored = sorted(scored_rows, key=lambda x: -x["taste_score"])

    safe_bets = [r for r in all_scored if r["category"] == "safe_bet"]
    wild_cards = [r for r in all_scored if r["category"] == "wild_card"]

    # Fallback: lower threshold if fewer than 2 safe bets
    if len(safe_bets) < 2:
        for r in all_scored:
            if r["category"] in ("wild_card", "skip") and r["taste_score"] >= SAFE_BET_FALLBACK_THRESHOLD:
                r["category"] = "safe_bet"
                safe_bets.append(r)
                wild_cards = [x for x in wild_cards if x is not r]
            if len(safe_bets) >= 2:
                break

    # Fallback: no wild cards — take highest-scoring skips with ≥1 negative keyword signal
    if not wild_cards:
        candidates_with_neg = [
            r for r in all_scored
            if r["category"] == "skip"
            and any(pts < 0 for kw, pts in [(k, v) for k, v in NEGATIVE_KEYWORDS.items() if k in (r.get("tags_str") or "")])
        ]
        wild_cards = sorted(candidates_with_neg, key=lambda x: -x["taste_score"])[:2]
        for r in wild_cards:
            r["category"] = "wild_card"

    # Diversity pass: if all top-3 safe bets share the same primary genre combo,
    # swap the lowest-scoring one for the best candidate with a different combo.
    def _primary_combo(item: dict) -> Optional[frozenset]:
        genre_set = set(item.get("genres", []))
        best = max(
            ((combo, bonus) for combo, bonus, _ in COMBO_BONUSES if combo.issubset(genre_set)),
            key=lambda x: x[1],
            default=(None, 0),
        )
        return best[0]

    if len(safe_bets) >= 3:
        dominant = _primary_combo(safe_bets[0])
        if dominant and all(_primary_combo(r) == dominant for r in safe_bets[:3]):
            safe_bet_titles = {r["title"] for r in safe_bets}
            for candidate in all_scored:
                if (
                    candidate["title"] not in safe_bet_titles
                    and _primary_combo(candidate) != dominant
                    and candidate["taste_score"] >= WILD_CARD_THRESHOLD
                ):
                    safe_bets[2] = candidate
                    candidate["category"] = "safe_bet"
                    wild_cards = [r for r in wild_cards if r["title"] != candidate["title"]]
                    break

    already_shown = {r["title"] for r in safe_bets[:3] + wild_cards[:2]}
    recent_picks = [
        r for r in all_scored
        if _year_int(r.get("year")) >= RECENT_ERA_MIN_YEAR
        and r["taste_score"] >= WILD_CARD_THRESHOLD
        and r["title"] not in already_shown
    ][:3]

    return {
        "safe_bets": safe_bets[:3],
        "wild_cards": wild_cards[:2],
        "recent_picks": recent_picks,
        "all_scored": all_scored,
    }
