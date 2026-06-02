#!/usr/bin/env python3
"""Export latest recommendations to the static GitHub Pages site."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS_DATA = ROOT / "docs" / "data"


@dataclass
class FilmsPrefs:
    era_min_year: Optional[int] = None
    era_max_year: Optional[int] = None
    exclude_famous: bool = False
    famous_votes_threshold: int = 500_000
    mood: str = "any"
    max_runtime: Optional[int] = None


@dataclass
class ShowsPrefs:
    era_min_year: Optional[int] = None
    status_filter: str = "any"
    max_episodes: Optional[int] = None
    exclude_famous: bool = False
    famous_votes_threshold: int = 200_000
    mood: str = "any"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _parse_film_genres(raw) -> list[str]:
    if not raw or pd.isna(raw):
        return []
    return [g.strip() for g in str(raw).split(",") if g.strip()]


def _parse_show_genres(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(g).strip().title() for g in raw if str(g).strip()]
    if not raw or pd.isna(raw):
        return []
    return [g.strip().title() for g in str(raw).split("|") if g.strip()]


def _films_taste_profile() -> dict[str, float]:
    ratings_path = ROOT / "films" / "my_ratings.csv"
    if not ratings_path.exists():
        return {}
    df = pd.read_csv(ratings_path)
    rows = []
    for _, row in df.iterrows():
        rating = row.get("Your Rating")
        if pd.isna(rating):
            continue
        for genre in _parse_film_genres(row.get("Genres", "")):
            rows.append({"genre": genre, "rating": float(rating)})
    if not rows:
        return {}
    out = pd.DataFrame(rows).groupby("genre")["rating"].mean().sort_values(ascending=False)
    return {genre: round(float(rating), 2) for genre, rating in out.items()}


def _shows_taste_profile(my_tv_ratings: dict) -> dict[str, float]:
    meta_path = ROOT / "shows" / "data" / "shows_meta.parquet"
    if not meta_path.exists():
        return {}
    meta = pd.read_parquet(meta_path, columns=["slug", "genres"])
    rows = []
    for slug, rating in my_tv_ratings.items():
        match = meta[meta["slug"] == slug]
        if match.empty:
            continue
        for genre in _parse_show_genres(match.iloc[0]["genres"]):
            rows.append({"genre": genre, "rating": float(rating)})
    if not rows:
        return {}
    out = pd.DataFrame(rows).groupby("genre")["rating"].mean().sort_values(ascending=False)
    return {genre: round(float(rating), 2) for genre, rating in out.items()}


def _mode_payload(
    mode: str,
    variant: str,
    label: str,
    results: dict,
    taste_profile: dict[str, float],
) -> dict:
    return {
        "status": "ok",
        "mode": mode,
        "variant": variant,
        "label": label,
        "date": datetime.now().date().isoformat(),
        "n_neighbours": results.get("n_neighbours", 0),
        "run_time_s": results.get("run_time_s", 0),
        "session_filters": results.get("session_filters", "none") or "none",
        "safe_bets": results.get("safe_bets", []),
        "wild_cards": results.get("wild_cards", []),
        "recent_picks": results.get("recent_picks", []),
        "all_scored": results.get("all_scored", []),
        "taste_profile": taste_profile,
    }


def _error_payload(mode: str, exc: Exception) -> dict:
    message = str(exc).replace(str(ROOT), ".")
    return {
        "status": "error",
        "mode": mode,
        "date": datetime.now().date().isoformat(),
        "error": message,
        "safe_bets": [],
        "wild_cards": [],
        "all_scored": [],
        "taste_profile": {},
    }


def build_payload() -> dict:
    sys.path.insert(0, str(ROOT))
    from app.pipeline import get_my_tv_ratings, run_films, run_shows

    exported_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": 1,
        "exported_at": exported_at,
        "source": "local export",
        "variants": [
            {"key": "all", "label": "All"},
            {"key": "post_2020", "label": "Post-2020"},
        ],
        "films": {"variants": {}},
        "shows": {"variants": {}},
    }

    film_profile = _films_taste_profile()
    try:
        show_profile = _shows_taste_profile(get_my_tv_ratings())
    except Exception:
        show_profile = {}

    film_variants = (
        ("all", "All", FilmsPrefs()),
        ("post_2020", "Post-2020", FilmsPrefs(era_min_year=2020)),
    )
    for key, label, prefs in film_variants:
        try:
            film_results = run_films(prefs)
            payload["films"]["variants"][key] = _mode_payload(
                "films", key, label, film_results, film_profile
            )
        except Exception as exc:
            payload["films"]["variants"][key] = _error_payload("films", exc)

    show_variants = (
        ("all", "All", ShowsPrefs()),
        ("post_2020", "Post-2020", ShowsPrefs(era_min_year=2020)),
    )
    for key, label, prefs in show_variants:
        try:
            show_results = run_shows(prefs)
            payload["shows"]["variants"][key] = _mode_payload(
                "shows", key, label, show_results, show_profile
            )
        except Exception as exc:
            payload["shows"]["variants"][key] = _error_payload("shows", exc)

    return _json_safe(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static recommendations site data")
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS_DATA / "recs.json",
        help="Output JSON path (default: docs/data/recs.json)",
    )
    args = parser.parse_args()

    t0 = time.time()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {args.output.relative_to(ROOT)} in {time.time() - t0:.1f}s")
    for mode in ("films", "shows"):
        for key, data in payload[mode]["variants"].items():
            label = data.get("label", key)
            if data["status"] == "ok":
                recent = len(data.get("recent_picks", []))
                recent_str = f", {recent} recent picks" if recent else ""
                print(
                    f"  {mode} {label}: {len(data['safe_bets'])} safe bets, "
                    f"{len(data['wild_cards'])} wild cards{recent_str}, {len(data['all_scored'])} candidates"
                )
            else:
                print(f"  {mode} {label}: not ready - {data['error']}")


if __name__ == "__main__":
    main()
