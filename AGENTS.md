# Repository Guidelines

## Project Structure & Module Organization

This repository contains a personal film and TV recommendation system with two independent pipelines and two user-facing entry points.

- `films/src/`: MovieLens-based film pipeline (`ingest.py`, `match.py`, `candidates.py`, `scorer.py`, `recommend.py`, `session.py`).
- `shows/src/`: Trakt-based TV pipeline with the same module roles as `films/src/`.
- `app/`: Streamlit dashboard support code, including pipeline wiring, cached data loading, charts, and UI components.
- `app.py`: Streamlit dashboard entry point.
- `addon.py`: local Stremio addon server.
- `films/data/` and `shows/data/`: generated parquet files and recommendation JSON outputs; do not hand-edit generated data.
- `films/my_ratings.csv`, `films/.env`, and `shows/.env`: local/private inputs and credentials.

## Build, Test, and Development Commands

Use a virtual environment before installing dependencies.

```bash
pip install -r requirements.txt       # core CLI pipeline dependencies
pip install -r requirements_app.txt   # Streamlit dashboard and addon extras
```

Pipeline setup:

```bash
cd films && python src/ingest.py && python src/match.py
cd shows && python src/ingest.py && python src/match.py
```

Run recommendations and apps:

```bash
cd films && python src/recommend.py --no-interactive
cd shows && python src/recommend.py --no-interactive
streamlit run app.py
python addon.py --port 7000
```

## Coding Style & Naming Conventions

Write Python with 4-space indentation, descriptive snake_case names, and module-level constants for paths, thresholds, and scoring weights. Keep pipeline modules dependency-light and deterministic. Prefer `pathlib.Path` for filesystem work and pandas parquet/CSV APIs for data I/O. Existing `films/src` and `shows/src` modules share names, so be careful with `sys.path` and module cache behavior in `app/pipeline.py`.

## Testing Guidelines

There is no test suite yet. When adding tests, use `pytest`, place them under `tests/`, and name files `test_<module>.py`. Focus first on deterministic scoring, candidate filtering, path handling, and parser behavior. For now, validate changes by running the relevant CLI command with `--no-interactive` and, when touching UI or addon code, smoke-test `streamlit run app.py` or `python addon.py`.

## Commit & Pull Request Guidelines

Git history currently only shows an initial commit, so no strict convention is established. Use short, imperative commit messages such as `Add dashboard score breakdown` or `Fix show candidate filtering`. Pull requests should include a concise description, commands run, affected pipeline (`films`, `shows`, `app`, or `addon`), linked issues if any, and screenshots for dashboard-visible changes.

## Security & Configuration Tips

Do not commit API keys, `.env` files, personal ratings exports, generated datasets, or recommendation outputs containing private taste data. Keep `OMDB_API_KEY` in `films/.env` and `TRAKT_CLIENT_ID` in `shows/.env`.
