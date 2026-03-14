"""
Main entry point — generate tonight's TV show recommendations.

Usage:
    python src/recommend.py                 # interactive questions then picks
    python src/recommend.py --no-interactive   # skip questions, use defaults
    python src/recommend.py --debug         # full scored candidate list
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MY_NEIGHBOURS = DATA_DIR / "my_neighbours.parquet"
USER_RATINGS = DATA_DIR / "user_ratings.parquet"


def check_prereqs():
    errors = []
    if not USER_RATINGS.exists():
        errors.append("  ✗ User ratings missing — run: python src/ingest.py")
    if not MY_NEIGHBOURS.exists():
        errors.append("  ✗ Neighbours data missing — run: python src/match.py")
    if errors:
        print("Setup incomplete:\n" + "\n".join(errors))
        sys.exit(1)


def get_session_preferences():
    """Interactive prompt — ask five questions and return a SessionPrefs."""
    sys.path.insert(0, str(Path(__file__).parent))
    from session import SessionPrefs

    prefs = SessionPrefs()

    width = 50
    line = "━" * width
    print(f"\n{line}")
    print("  What are you in the mood for tonight? (TV)")
    print(line)
    print("  (Press Enter to skip any question)\n")

    # ── Question 1: Era ──────────────────────────────────────────────────────
    print("1. Any era preference?")
    print("   [1] Recent only (2015+)")
    print("   [2] Post-2000 only")
    print("   [3] No preference")
    era_input = input("   → ").strip()
    if era_input == "1":
        prefs.era_min_year = 2015
    elif era_input == "2":
        prefs.era_min_year = 2000

    # ── Question 2: Status ───────────────────────────────────────────────────
    print("\n2. Finished or ongoing?")
    print("   [1] Finished only — I want a complete story")
    print("   [2] Ongoing only — I want something current")
    print("   [3] No preference")
    status_input = input("   → ").strip()
    if status_input == "1":
        prefs.status_filter = "ended"
    elif status_input == "2":
        prefs.status_filter = "ongoing"

    # ── Question 3: Episode commitment ──────────────────────────────────────
    print("\n3. How much of a commitment?")
    print("   [1] Short run (≤15 episodes total)")
    print("   [2] Medium run (≤50 episodes total)")
    print("   [3] Happy to dive into something long")
    eps_input = input("   → ").strip()
    if eps_input == "1":
        prefs.max_episodes = 15
    elif eps_input == "2":
        prefs.max_episodes = 50

    # ── Question 4: Obscurity ────────────────────────────────────────────────
    print("\n4. How well-known?")
    print("   [1] Hidden gems only (avoid anything everyone knows)")
    print("   [2] No mainstream hits (filter out very widely-seen shows)")
    print("   [3] No preference")
    obs_input = input("   → ").strip()
    if obs_input == "1":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 50_000
    elif obs_input == "2":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 200_000

    # ── Question 5: Mood ─────────────────────────────────────────────────────
    print("\n5. What mood?")
    print("   [1] Gripping / tense")
    print("   [2] Thought-provoking")
    print("   [3] Lighter")
    print("   [4] No preference")
    mood_input = input("   → ").strip()
    if mood_input == "1":
        prefs.mood = "tense"
    elif mood_input == "2":
        prefs.mood = "thoughtful"
    elif mood_input == "3":
        prefs.mood = "lighter"

    print(f"\n{'━' * width}")
    print("  Finding picks...\n")
    return prefs


def fmt_genres(genres: list) -> str:
    return " · ".join(g.title() for g in genres) if genres else "Unknown"


def fmt_year(year) -> str:
    if year is None:
        return ""
    try:
        return str(int(year))
    except (ValueError, TypeError):
        return ""


def fmt_active_filters(prefs) -> str:
    parts = []
    if prefs.era_min_year:
        parts.append(f"Era: post-{prefs.era_min_year}")
    if prefs.status_filter != "any":
        parts.append(f"Status: {prefs.status_filter}")
    if prefs.max_episodes:
        parts.append(f"≤{prefs.max_episodes} episodes")
    if prefs.exclude_famous:
        threshold_k = prefs.famous_votes_threshold // 1000
        parts.append(f"Familiar titles filtered (<{threshold_k}k votes)")
    if prefs.mood != "any":
        parts.append(f"Mood: {prefs.mood}")
    return " · ".join(parts)


def fmt_status(status: str, aired_episodes) -> str:
    status_str = {
        "ended": "Ended",
        "canceled": "Cancelled",
        "returning series": "Ongoing",
        "in production": "In Production",
    }.get((status or "").lower(), status or "Unknown")
    if aired_episodes:
        return f"{status_str} · {int(aired_episodes)} episodes"
    return status_str


def print_recommendations(results: dict, n_neighbours: int, run_time: float, prefs):
    safe_bets = results["safe_bets"]
    wild_cards = results["wild_cards"]
    today = date.today().isoformat()

    width = 50
    line = "━" * width

    print(f"\n{line}")
    print(f"  TONIGHT'S PICKS (TV)  —  {today}")
    filter_str = fmt_active_filters(prefs)
    if filter_str:
        print(f"  [{filter_str}]")
    print(line)

    if safe_bets:
        print("\n✓ SAFE BETS\n")
        for show in safe_bets:
            year_str = f" ({fmt_year(show['year'])})" if fmt_year(show["year"]) else ""
            print(f"  {show['title']}{year_str} — {fmt_genres(show['genres'])}")
            print(f"  {fmt_status(show['status'], show['aired_episodes'])}")
            print(
                f"  Taste score: {show['taste_score']:.0f}/100  |  "
                f"Recommended by {show['recommended_by']}/{n_neighbours} neighbours "
                f"(avg {show['neighbour_score']:.1f}★)"
            )
            for reason in show["reasons"]:
                print(f"    ↳ {reason}")
            print()
    else:
        print("\n  No safe bets found — try relaxing filters or lowering SAFE_BET_THRESHOLD in scorer.py")

    if wild_cards:
        print("◆ WILD CARDS\n")
        for show in wild_cards:
            year_str = f" ({fmt_year(show['year'])})" if fmt_year(show["year"]) else ""
            print(f"  {show['title']}{year_str} — {fmt_genres(show['genres'])}")
            print(f"  {fmt_status(show['status'], show['aired_episodes'])}")
            print(
                f"  Taste score: {show['taste_score']:.0f}/100  |  "
                f"Recommended by {show['recommended_by']}/{n_neighbours} neighbours "
                f"(avg {show['neighbour_score']:.1f}★)"
            )
            for reason in show["reasons"]:
                print(f"    ↳ {reason}")
            print()
    else:
        print("  No wild cards found.\n")

    print(line)
    print(f"  {n_neighbours} neighbours · Trakt.tv · Run time: {run_time:.1f}s")
    print()
    print("  Watched something new? Update MY_TV_RATINGS in src/match.py, then:")
    print("  python src/match.py")
    print(line + "\n")


def print_debug(results: dict, n_neighbours: int):
    all_scored = results.get("all_scored", [])
    print(f"\n{'─'*70}")
    print(f"DEBUG — all {len(all_scored)} candidates (sorted by taste_score)")
    print(f"{'─'*70}")
    print(f"{'Score':>6}  {'Cat':>14}  {'Nbrs':>5}  {'Title'}")
    print("─" * 70)
    for show in all_scored:
        cat = show["category"]
        print(
            f"{show['taste_score']:>6.1f}  {cat:>14}  "
            f"{show['recommended_by']:>4}/{n_neighbours}  "
            f"{show['title']} ({fmt_year(show['year'])})"
        )
        for reason in show["reasons"][:3]:
            print(f"         {reason}")
    print()


def save_json(results: dict, n_neighbours: int, run_time: float, prefs):
    today = date.today().isoformat()
    output_path = DATA_DIR / f"tv_recommendations_{today}.json"

    payload = {
        "date": today,
        "n_neighbours": n_neighbours,
        "run_time_s": round(run_time, 2),
        "session_filters": fmt_active_filters(prefs) or "none",
        "safe_bets": results["safe_bets"],
        "wild_cards": results["wild_cards"],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved to {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="TV show recommendation engine")
    parser.add_argument("--debug", action="store_true", help="Print full scored candidate list")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip questions and use default preferences",
    )
    args = parser.parse_args()

    check_prereqs()

    sys.path.insert(0, str(Path(__file__).parent))
    from session import SessionPrefs
    from match import MY_TV_RATINGS
    from candidates import generate_candidates
    from scorer import score_and_select
    import pandas as pd

    if args.no_interactive:
        prefs = SessionPrefs()
    else:
        prefs = get_session_preferences()

    t_start = time.time()

    candidates = generate_candidates(MY_TV_RATINGS, prefs)

    if len(candidates) == 0:
        print(
            "\nNo candidates found. Possible causes:\n"
            "  • Session filters too restrictive — try relaxing era/status/episode filters\n"
            "  • Neighbour quality too low — try re-running match.py\n"
            "  • All recommended shows are in the already-seen list\n"
            "  • Filter thresholds too strict (see candidates.py constants)"
        )
        sys.exit(1)

    results = score_and_select(candidates, mood=prefs.mood)

    run_time = time.time() - t_start

    neighbours = pd.read_parquet(MY_NEIGHBOURS)
    n_neighbours = len(neighbours)

    print_recommendations(results, n_neighbours, run_time, prefs)

    if args.debug:
        print_debug(results, n_neighbours)

    save_json(results, n_neighbours, run_time, prefs)


if __name__ == "__main__":
    main()
