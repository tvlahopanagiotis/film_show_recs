"""
One-time script: split my_ratings.csv into films and TV shows.

Usage:
    python split_ratings.py

Reads:  imdb_ratings_new.csv  (original IMDb export)
Writes: films/my_ratings.csv
        shows/my_tv_ratings.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SOURCE = ROOT / "imdb_ratings_new.csv"

if not SOURCE.exists():
    raise SystemExit(f"Source file not found: {SOURCE}")

df = pd.read_csv(SOURCE)

TV_TYPES   = {"tvSeries", "tvMiniSeries"}
FILM_TYPES = {"movie", "video", "short", "tvMovie"}

tv    = df[df["Title Type"].isin(TV_TYPES)]
films = df[df["Title Type"].isin(FILM_TYPES)]
other = df[~df["Title Type"].isin(TV_TYPES | FILM_TYPES)]

films.to_csv(ROOT / "films" / "my_ratings.csv", index=False)
tv.to_csv(ROOT / "shows" / "my_tv_ratings.csv", index=False)

print(f"{len(films)} films  →  films/my_ratings.csv")
print(f"{len(tv)} TV shows  →  shows/my_tv_ratings.csv")
if len(other):
    print(f"{len(other)} other entries skipped (types: {other['Title Type'].unique().tolist()})")
print(f"Total: {len(df)} ratings processed")
