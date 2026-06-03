"""
Main entry point — generate tonight's film recommendations.

Usage:
    python src/recommend.py                # interactive questions then picks
    python src/recommend.py --no-interactive   # skip questions, use defaults
    python src/recommend.py --debug        # full scored candidate list
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_RATINGS = DATA_DIR / "processed_ratings.parquet"
MY_NEIGHBOURS = DATA_DIR / "my_neighbours.parquet"
MY_RATINGS_PATH = Path(__file__).parent.parent / "my_ratings.csv"


def check_prereqs():
    errors = []
    if not PROCESSED_RATINGS.exists():
        errors.append("  ✗ Preprocessed data missing — run: python src/ingest.py")
    if not MY_NEIGHBOURS.exists():
        errors.append("  ✗ Neighbours data missing — run: python src/match.py")

    ratings_ok = MY_RATINGS_PATH.exists() or (Path(__file__).parent.parent / "imdb_ratings_new.csv").exists()
    if not ratings_ok:
        errors.append(
            "  ✗ My ratings not found — export from IMDb and save as my_ratings.csv"
        )

    if errors:
        print("Setup incomplete:\n" + "\n".join(errors))
        sys.exit(1)


def get_session_preferences():
    """Interactive prompt — ask five quick questions and return a SessionPrefs."""
    sys.path.insert(0, str(Path(__file__).parent))
    from session import SessionPrefs

    prefs = SessionPrefs()

    width = 50
    line = "━" * width
    print(f"\n{line}")
    print("  What are you in the mood for tonight?")
    print(line)
    print("  (Press Enter to skip any question)\n")

    # ── Question 1: Era ──────────────────────────────
    print("1. Any era preference?")
    print("   [1] Post-2000 only")
    print("   [2] Post-2010 only")
    print("   [3] Classics only (pre-1980)")
    print("   [4] No preference")
    print("   (Note: MovieLens data ends at 2019 — post-2020 films use the Recent Picks section)")
    era_input = input("   → ").strip()
    if era_input == "1":
        prefs.era_min_year = 2000
    elif era_input == "2":
        prefs.era_min_year = 2010
    elif era_input == "3":
        prefs.era_max_year = 1980

    # ── Question 2: Obscurity ────────────────────────
    print("\n2. How well-known?")
    print("   [1] Hidden gems only (avoid anything everyone knows)")
    print("   [2] No obvious blockbusters (filter out very widely-seen films)")
    print("   [3] No preference")
    obs_input = input("   → ").strip()
    if obs_input == "1":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 200_000
    elif obs_input == "2":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 500_000

    # ── Question 3: Mood ─────────────────────────────
    print("\n3. What mood?")
    print("   [1] Gripping / tense — I want to be on the edge of my seat")
    print("   [2] Thought-provoking — I want something that stays with me")
    print("   [3] Lighter — something engaging but not draining")
    print("   [4] No preference")
    mood_input = input("   → ").strip()
    if mood_input == "1":
        prefs.mood = "tense"
    elif mood_input == "2":
        prefs.mood = "thoughtful"
    elif mood_input == "3":
        prefs.mood = "lighter"

    # ── Question 4: Runtime ──────────────────────────
    print("\n4. How much time do you have?")
    print("   [1] Short (~90 min)")
    print("   [2] Standard (~2 hrs)")
    print("   [3] Happy to commit (2.5 hrs+)")
    print("   [4] No preference")
    runtime_input = input("   → ").strip()
    if runtime_input == "1":
        prefs.max_runtime = 100
    elif runtime_input == "2":
        prefs.max_runtime = 135

    print(f"\n{'━' * width}")
    print("  Finding picks...\n")
    return prefs


def fmt_genres(genres: list) -> str:
    return " · ".join(genres) if genres else "Unknown"


def fmt_year(year) -> str:
    if year is None:
        return ""
    try:
        return str(int(year))
    except (ValueError, TypeError):
        return ""


def fmt_active_filters(prefs) -> str:
    """Build a one-line summary of the active session filters."""
    parts = []
    if prefs.era_min_year and prefs.era_max_year:
        parts.append(f"Era: {prefs.era_min_year}–{prefs.era_max_year}")
    elif prefs.era_min_year:
        parts.append(f"Era: post-{prefs.era_min_year}")
    elif prefs.era_max_year:
        parts.append(f"Era: pre-{prefs.era_max_year + 1}")
    if prefs.exclude_famous:
        threshold_k = prefs.famous_votes_threshold // 1000
        parts.append(f"Familiar titles filtered (<{threshold_k}k votes)")
    if prefs.mood != "any":
        parts.append(f"Mood: {prefs.mood}")
    if prefs.max_runtime:
        parts.append(f"Runtime: ≤{prefs.max_runtime} min")
    return " · ".join(parts)


def _print_film_entry(film: dict, n_neighbours: int):
    year_str = f" ({fmt_year(film['year'])})" if fmt_year(film["year"]) else ""
    print(f"  {film['title']}{year_str} — {fmt_genres(film['genres'])}")
    print(
        f"  Taste score: {film['taste_score']:.0f}/100  |  "
        f"Recommended by {film['recommended_by']}/{n_neighbours} neighbours "
        f"(avg {film['neighbour_score']:.1f}★)"
    )
    for reason in film["reasons"]:
        print(f"    ↳ {reason}")
    print()


def print_recommendations(results: dict, n_neighbours: int, run_time: float, prefs):
    safe_bets = results["safe_bets"]
    wild_cards = results["wild_cards"]
    recent_picks = results.get("recent_picks", [])
    today = date.today().isoformat()

    width = 50
    line = "━" * width

    print(f"\n{line}")
    print(f"  TONIGHT'S PICKS  —  {today}")
    filter_str = fmt_active_filters(prefs)
    if filter_str:
        print(f"  [{filter_str}]")
    print(line)

    if safe_bets:
        print("\n✓ SAFE BETS\n")
        for film in safe_bets:
            _print_film_entry(film, n_neighbours)
    else:
        print("\n  No safe bets found — try lowering SAFE_BET_THRESHOLD in scorer.py")

    if wild_cards:
        print("◆ WILD CARDS\n")
        for film in wild_cards:
            _print_film_entry(film, n_neighbours)
    else:
        print("  No wild cards found.\n")

    if recent_picks:
        print("◈ RECENT PICKS (2020+)\n")
        for film in recent_picks:
            _print_film_entry(film, n_neighbours)

    print(line)
    print(f"  {n_neighbours} neighbours · MovieLens ml-32m · Run time: {run_time:.1f}s")
    print()
    print("  Watched something? Export IMDb ratings, replace my_ratings.csv, then:")
    print("  python src/match.py")
    print(line + "\n")


def print_debug(results: dict, n_neighbours: int):
    all_scored = results.get("all_scored", [])
    print(f"\n{'─'*70}")
    print(f"DEBUG — all {len(all_scored)} candidates (sorted by taste_score)")
    print(f"{'─'*70}")
    print(f"{'Score':>6}  {'Cat':>9}  {'Nbrs':>5}  {'Title'}")
    print("─" * 70)
    for film in all_scored:
        cat = film["category"]
        print(
            f"{film['taste_score']:>6.1f}  {cat:>9}  "
            f"{film['recommended_by']:>4}/{n_neighbours}  "
            f"{film['title']} ({fmt_year(film['year'])})"
        )
        for reason in film["reasons"][:3]:
            print(f"         {reason}")
    print()


def save_json(results: dict, n_neighbours: int, run_time: float, prefs):
    today = date.today().isoformat()
    output_path = DATA_DIR / f"recommendations_{today}.json"

    payload = {
        "date": today,
        "n_neighbours": n_neighbours,
        "run_time_s": round(run_time, 2),
        "session_filters": fmt_active_filters(prefs) or "none",
        "safe_bets": results["safe_bets"],
        "wild_cards": results["wild_cards"],
        "recent_picks": results.get("recent_picks", []),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved to {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Film recommendation engine")
    parser.add_argument("--debug", action="store_true", help="Print full scored candidate list")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip questions and use default preferences (useful for scripting)",
    )
    args = parser.parse_args()

    check_prereqs()

    sys.path.insert(0, str(Path(__file__).parent))
    from session import SessionPrefs
    from match import load_my_ratings, get_era_weighted_neighbours
    from candidates import generate_candidates
    from scorer import score_candidates
    import pandas as pd

    if args.no_interactive:
        prefs = SessionPrefs()
    else:
        prefs = get_session_preferences()

    t_start = time.time()

    my_ratings = load_my_ratings()

    # Use era-weighted neighbours for the session when an era filter is active,
    # so Pearson similarity is computed on era-relevant films only (not saved to disk)
    era_neighbours = None
    if prefs.era_min_year is not None:
        era_neighbours = get_era_weighted_neighbours(my_ratings, prefs.era_min_year)
        if era_neighbours is None:
            print("  Falling back to precomputed neighbours.")

    candidates = generate_candidates(my_ratings, prefs, neighbours=era_neighbours)

    if len(candidates) == 0:
        print(
            "\nNo candidates found. Possible causes:\n"
            "  • Session filters too restrictive — try relaxing era/type/runtime filters\n"
            "  • Neighbour quality too low — try re-running match.py\n"
            "  • All recommended films are in the already-seen list\n"
            "  • Filter thresholds too strict (see candidates.py constants)"
        )
        sys.exit(1)

    results = score_candidates(candidates, mood=prefs.mood)

    run_time = time.time() - t_start

    neighbours = pd.read_parquet(MY_NEIGHBOURS)
    n_neighbours = len(neighbours)

    print_recommendations(results, n_neighbours, run_time, prefs)

    if args.debug:
        print_debug(results, n_neighbours)

    save_json(results, n_neighbours, run_time, prefs)


if __name__ == "__main__":
    main()
