#!/usr/bin/env python3
"""
Personal Stremio addon — serves personalised film and show recommendations
as local Stremio catalogs, identified by IMDB IDs.

Usage:
    python addon.py              # starts on port 7000
    python addon.py --port 7777

Install in Stremio:
    Addons → search icon → "Add addon from URL"
    Same machine:  http://localhost:7000/manifest.json
    Phone (LAN):   http://<your-mac-ip>:7000/manifest.json
                   (find IP with: ipconfig getifaddr en0)

Stremio fetches posters and streaming links automatically via IMDB IDs.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Prefs dataclasses (mirrors app.py — no streamlit needed here) ─────────────


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
    era_max_year: Optional[int] = None
    status_filter: str = "any"
    max_episodes: Optional[int] = None
    exclude_famous: bool = False
    famous_votes_threshold: int = 200_000
    mood: str = "any"


# ── IMDB ID helpers ───────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
SHOWS_META_PATH = ROOT / "shows" / "data" / "shows_meta.parquet"


def _fmt_film_id(raw) -> Optional[str]:
    """Convert numeric imdbId → 'tt0123456' (zero-padded to 7 digits)."""
    if raw is None:
        return None
    try:
        n = int(raw)
        return f"tt{n:07d}"
    except (ValueError, TypeError):
        return None


def _load_slug_to_imdb() -> dict[str, str]:
    """Build slug → imdb_id lookup from shows_meta.parquet."""
    try:
        df = pd.read_parquet(SHOWS_META_PATH, columns=["slug", "imdb_id"])
        return {
            row["slug"]: row["imdb_id"]
            for _, row in df.iterrows()
            if row["imdb_id"] and str(row["imdb_id"]).startswith("tt")
        }
    except FileNotFoundError:
        print(f"  Warning: {SHOWS_META_PATH} not found — shows catalog will be empty.")
        return {}


# ── Stremio manifest ──────────────────────────────────────────────────────────

MANIFEST = {
    "id": "local.personalrecs",
    "version": "1.0.0",
    "name": "My Recommendations",
    "description": "Personalised collaborative-filtering picks — films & shows",
    "types": ["movie", "series"],
    "catalogs": [
        {"type": "movie",  "id": "personal-films", "name": "My Film Picks"},
        {"type": "series", "id": "personal-shows", "name": "My Show Picks"},
    ],
    "resources": ["catalog"],
    "idPrefixes": ["tt"],
    "behaviorHints": {"adult": False, "p2p": False},
}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _build_catalogs() -> tuple[list[dict], list[dict]]:
    """Run both pipelines with default prefs and return (films_metas, shows_metas)."""
    from app.pipeline import run_films, run_shows

    # Films
    print("Running films pipeline...", end=" ", flush=True)
    try:
        film_results = run_films(FilmsPrefs())
        all_films = sorted(
            film_results["all_scored"], key=lambda x: x.get("taste_score", 0), reverse=True
        )
        films_metas = []
        for item in all_films:
            imdb_id = _fmt_film_id(item.get("imdbId"))
            if not imdb_id:
                continue
            films_metas.append({
                "id": imdb_id,
                "type": "movie",
                "name": item["title"],
                "poster": f"https://images.metahub.space/poster/medium/{imdb_id}/img",
            })
        print(f"done — {len(all_films)} candidates, {len(films_metas)} with IMDB ID")
    except Exception as e:
        print(f"ERROR: {e}")
        films_metas = []

    # Shows
    print("Running shows pipeline...", end=" ", flush=True)
    try:
        slug_to_imdb = _load_slug_to_imdb()
        show_results = run_shows(ShowsPrefs())
        all_shows = sorted(
            show_results["all_scored"], key=lambda x: x.get("taste_score", 0), reverse=True
        )
        shows_metas = []
        for item in all_shows:
            imdb_id = slug_to_imdb.get(item.get("slug", ""))
            if not imdb_id:
                continue
            shows_metas.append({
                "id": imdb_id,
                "type": "series",
                "name": item["title"],
                "poster": f"https://images.metahub.space/poster/medium/{imdb_id}/img",
            })
        print(f"done — {len(all_shows)} candidates, {len(shows_metas)} with IMDB ID")
    except Exception as e:
        print(f"ERROR: {e}")
        shows_metas = []

    return films_metas, shows_metas


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    films_metas: list[dict] = []
    shows_metas: list[dict] = []

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/manifest.json":
            self._send_json(MANIFEST)
        elif path == "/catalog/movie/personal-films.json":
            self._send_json({"metas": self.films_metas})
        elif path == "/catalog/series/personal-shows.json":
            self._send_json({"metas": self.shows_metas})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Only log non-manifest requests to avoid noise
        path = args[0].split()[1] if args else ""
        if "/manifest" not in path:
            print(f"  {self.address_string()} {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _find_free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise the next available one."""
    import socket
    for port in range(preferred, preferred + 10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found near {preferred}")


def main():
    parser = argparse.ArgumentParser(description="Personal Stremio addon server")
    parser.add_argument("--port", type=int, default=7000, help="Port to listen on (default: 7000)")
    args = parser.parse_args()

    print("Building catalogs…")
    films_metas, shows_metas = _build_catalogs()

    # Inject catalog data into handler class
    _Handler.films_metas = films_metas
    _Handler.shows_metas = shows_metas

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use — using {port} instead.")

    server = HTTPServer(("0.0.0.0", port), _Handler)

    try:
        local_ip = _get_local_ip()
    except Exception:
        local_ip = "<your-mac-ip>"

    print(f"\nAddon running at:")
    print(f"  http://localhost:{port}/manifest.json          (same machine)")
    print(f"  http://{local_ip}:{port}/manifest.json   (LAN / phone)")
    print(f"\nIn Stremio: Addons → search icon → 'Add addon from URL' → paste one of the above")
    print(f"Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def _get_local_ip() -> str:
    """Best-effort local LAN IP detection."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


if __name__ == "__main__":
    main()
