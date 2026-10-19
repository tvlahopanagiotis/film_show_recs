#!/usr/bin/env python3
"""
Offline evaluation: how well does each pipeline predict ratings it hasn't seen?

K-fold cross-validation over your own ratings. For each fold, the held-out
titles are hidden, neighbours are recomputed from the remaining ratings, and
the full candidate + scoring pipeline runs. Then we check where the hidden
titles landed:

  liked     = hidden titles you rated >= LIKED_MIN   (should rank high)
  disliked  = hidden titles you rated <= DISLIKED_MAX (should rank low)

Metrics (pooled across folds, so every rating is held out exactly once):
  reach     share of hidden titles that make it into the candidate pool at all
  AUC       chance a random liked title outranks a random disliked one
            (0.5 = coin flip, 1.0 = perfect). Titles outside the pool rank last.
            "CF" ranks by the collaborative diversity_score alone; "final" by
            the scorer's taste_score, i.e. what actually decides the picks.

Films: only ratings that bridge to MovieLens are testable; TMDB recent picks
are content-based and skipped. Nothing in data/ is overwritten.

Usage (from recs/):
  python scripts/evaluate.py                # both pipelines, 5 folds
  python scripts/evaluate.py --only shows --folds 10
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TQDM_DISABLE", "1")

from app.pipeline import _activate_films, _activate_shows  # noqa: E402

LIKED_MIN = 8
DISLIKED_MAX = 5
TMP_DIR = Path(tempfile.mkdtemp(prefix="recs_eval_"))


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _folds(keys: list, k: int, seed: int) -> list[set]:
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    return [set(keys[i::k]) for i in range(k)]


def _auc(liked: list[float], disliked: list[float]) -> float:
    """Pairwise AUC; missing scores (not in pool) rank below every real score."""
    lo = float("-inf")
    liked = [lo if pd.isna(s) else s for s in liked]
    disliked = [lo if pd.isna(s) else s for s in disliked]
    if not liked or not disliked:
        return float("nan")
    wins = sum((l > d) + 0.5 * (l == d) for l in liked for d in disliked)
    return wins / (len(liked) * len(disliked))


def _summarise(name: str, rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    liked = df[df["rating"] >= LIKED_MIN]
    disliked = df[df["rating"] <= DISLIKED_MAX]
    return {
        "pipeline": name,
        "tested": len(df),
        "liked": len(liked),
        "disliked": len(disliked),
        "reach_liked": liked["cf"].notna().mean(),
        "reach_disliked": disliked["cf"].notna().mean(),
        "auc_cf": _auc(liked["cf"].tolist(), disliked["cf"].tolist()),
        "auc_final": _auc(liked["final"].tolist(), disliked["final"].tolist()),
    }


def _none_if_nan(x):
    return None if x is None or pd.isna(x) else float(x)


# ── Films ─────────────────────────────────────────────────────────────────────

def evaluate_films(k: int, seed: int) -> dict:
    _activate_films()
    import match
    import candidates
    from scorer import score_candidates

    match.MY_NEIGHBOURS = TMP_DIR / "films_neighbours.parquet"
    candidates.CANDIDATES_OUT = TMP_DIR / "films_candidates.parquet"
    candidates.TMDB_RECENT = TMP_DIR / "missing.parquet"  # skip content-based recent picks
    candidates.ALREADY_SEEN = set()                      # held-out titles must be allowed back in
    candidates.TOP_N_CANDIDATES = 10**9                  # rank the whole pool, not just the top 80

    with _quiet():
        my_ratings = match.load_my_ratings()

    rows = []
    for i, held_out in enumerate(_folds(list(my_ratings), k, seed), 1):
        t0 = time.time()
        train = {m: r for m, r in my_ratings.items() if m not in held_out}
        with _quiet():
            neighbours = match.find_neighbours(train)
            cands = candidates.generate_candidates(train, neighbours=neighbours)
            scored = score_candidates(cands)["all_scored"]

        cands = cands[cands["movieId"].notna()]
        cf = dict(zip(cands["movieId"].astype(int), cands["diversity_score"]))
        imdb_of = dict(zip(cands["movieId"].astype(int), cands["imdbId"]))
        final = {s["imdbId"]: s["taste_score"] for s in scored if s.get("imdbId") is not None}
        for m in held_out:
            rows.append({
                "rating": my_ratings[m],
                "cf": _none_if_nan(cf.get(m)),
                "final": _none_if_nan(final.get(imdb_of.get(m))),
            })
        print(f"  films fold {i}/{k} ({time.time() - t0:.0f}s)")

    return _summarise("films", rows)


# ── Shows ─────────────────────────────────────────────────────────────────────

def evaluate_shows(k: int, seed: int) -> dict:
    _activate_shows()
    import match
    import candidates
    from scorer import score_and_select

    match.MY_NEIGHBOURS = TMP_DIR / "shows_neighbours.parquet"
    match.TV_ALREADY_SEEN = set()                        # read by generate_candidates at call time
    candidates.CANDIDATES_OUT = TMP_DIR / "shows_candidates.parquet"
    candidates.TOP_N_CANDIDATES = 10**9

    with _quiet():
        my_ratings = match.load_my_ratings()

    rows = []
    for i, held_out in enumerate(_folds(list(my_ratings), k, seed), 1):
        train = {s: r for s, r in my_ratings.items() if s not in held_out}
        with _quiet():
            neighbours = match.find_neighbours(train)
            cands = candidates.generate_candidates(train, neighbours=neighbours)
            scored = score_and_select(cands)["all_scored"]

        cf = dict(zip(cands["slug"], cands["diversity_score"]))
        final = {s["slug"]: s["taste_score"] for s in scored}
        for s in held_out:
            rows.append({
                "rating": my_ratings[s],
                "cf": _none_if_nan(cf.get(s)),
                "final": _none_if_nan(final.get(s)),
            })
    print(f"  shows: {k} folds done")

    return _summarise("shows", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=["films", "shows"])
    args = parser.parse_args()

    results = []
    if args.only in (None, "shows"):
        results.append(evaluate_shows(args.folds, args.seed))
    if args.only in (None, "films"):
        results.append(evaluate_films(args.folds, args.seed))

    print(f"\n{args.folds}-fold evaluation (liked >= {LIKED_MIN}, disliked <= {DISLIKED_MAX})\n")
    print(f"{'':8}{'tested':>8}{'liked':>7}{'disl.':>7}{'reach L':>9}{'reach D':>9}{'AUC CF':>8}{'AUC final':>11}")
    for r in results:
        print(
            f"{r['pipeline']:8}{r['tested']:>8}{r['liked']:>7}{r['disliked']:>7}"
            f"{r['reach_liked']:>9.0%}{r['reach_disliked']:>9.0%}"
            f"{r['auc_cf']:>8.2f}{r['auc_final']:>11.2f}"
        )


if __name__ == "__main__":
    main()
