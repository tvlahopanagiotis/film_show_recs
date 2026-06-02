"""
Recommendation Dashboard — entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from app.charts import candidate_scatter, taste_profile_chart
from app.components import recommendation_card
from app.data_loader import (
    FILMS_DATA,
    SHOWS_DATA,
    films_genre_stats,
    latest_json,
    load_my_ratings,
    load_shows_meta,
    shows_genre_stats,
)
from app.pipeline import get_my_tv_ratings, run_films, run_shows

# ── Local SessionPrefs mirrors ─────────────────────────────────────────────────
# These replicate the fields of films/src/session.py and shows/src/session.py
# so we can construct prefs objects without sys.path games in the UI layer.


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


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Recs Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────

st.session_state.setdefault("mode", "films")
st.session_state.setdefault("results", None)
st.session_state.setdefault("selected_title", None)

# Cold-start: load latest JSON if no run has happened yet
if st.session_state.results is None:
    _mode = st.session_state.mode
    if _mode == "films":
        cold = latest_json(FILMS_DATA, "recommendations_")
    else:
        cold = latest_json(SHOWS_DATA, "tv_recommendations_")
    if cold:
        # Inject mode and n_neighbours if present in JSON
        cold["mode"] = _mode
        cold.setdefault("n_neighbours", 0)
        cold.setdefault("run_time_s", 0)
        cold.setdefault("session_filters", "")
        cold.setdefault("recent_picks", [])
        cold["all_scored"] = cold.get("safe_bets", []) + cold.get("wild_cards", [])
        st.session_state.results = cold


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 Recs")

    mode = st.radio("Mode", ["Films", "Shows"], horizontal=True)
    mode_key = mode.lower()
    if mode_key != st.session_state.mode:
        st.session_state.mode = mode_key
        st.session_state.results = None
        st.session_state.selected_title = None
        # Load cold-start for new mode
        if mode_key == "films":
            cold = latest_json(FILMS_DATA, "recommendations_")
        else:
            cold = latest_json(SHOWS_DATA, "tv_recommendations_")
        if cold:
            cold["mode"] = mode_key
            cold.setdefault("n_neighbours", 0)
            cold.setdefault("run_time_s", 0)
            cold.setdefault("session_filters", "")
            cold.setdefault("recent_picks", [])
            cold["all_scored"] = cold.get("safe_bets", []) + cold.get("wild_cards", [])
            st.session_state.results = cold

    st.divider()
    st.markdown("**Filters**")

    if mode_key == "films":
        era_opts = ["All", "Classic 1950–75", "Golden 1976–99", "Post-2000", "Recent 2010+"]
        era_sel = st.selectbox("Era", era_opts, key="films_era")
        st.caption("Post-2020 films aren't in MovieLens — see Recent Picks below.")

        fame_opts = ["No filter", "<500k votes", "<200k (hidden gems)"]
        fame_sel = st.selectbox("Famousness", fame_opts, key="films_fame")

        mood_opts = ["Any", "Tense", "Thoughtful", "Lighter"]
        mood_sel = st.selectbox("Mood", mood_opts, key="films_mood")

        runtime_opts = ["Any", "Under 100 min", "Under 135 min"]
        runtime_sel = st.selectbox("Runtime", runtime_opts, key="films_runtime")

    else:  # shows
        era_opts_s = ["All", "Post-2020", "Recent 2015+", "Post-2000"]
        era_sel_s = st.selectbox("Era", era_opts_s, key="shows_era")

        status_opts = ["Any", "Finished only", "Ongoing only"]
        status_sel = st.selectbox("Status", status_opts, key="shows_status")

        eps_opts = ["Any", "Short ≤15", "Medium ≤50"]
        eps_sel = st.selectbox("Episodes", eps_opts, key="shows_eps")

        fame_opts_s = ["No filter", "<200k votes", "<50k (hidden gems)"]
        fame_sel_s = st.selectbox("Famousness", fame_opts_s, key="shows_fame")

        mood_opts_s = ["Any", "Tense", "Thoughtful", "Lighter"]
        mood_sel_s = st.selectbox("Mood", mood_opts_s, key="shows_mood")

    st.divider()

    run_clicked = st.button("Get Recommendations", type="primary", use_container_width=True)

    # Last-run metadata
    if st.session_state.results:
        r = st.session_state.results
        st.divider()
        st.markdown("**Last run**")
        st.caption(f"Neighbours: {r.get('n_neighbours', '—')}")
        st.caption(f"Run time: {r.get('run_time_s', '—')}s")
        filters = r.get("session_filters", "")
        if filters and filters != "none":
            st.caption(f"Filters: {filters}")


# ── Build prefs and trigger pipeline ──────────────────────────────────────────

def _build_films_prefs() -> FilmsPrefs:
    prefs = FilmsPrefs()
    era = st.session_state.get("films_era", "All")
    if era == "Classic 1950–75":
        prefs.era_min_year = 1950
        prefs.era_max_year = 1975
    elif era == "Golden 1976–99":
        prefs.era_min_year = 1976
        prefs.era_max_year = 1999
    elif era == "Post-2000":
        prefs.era_min_year = 2000
    elif era == "Recent 2010+":
        prefs.era_min_year = 2010

    fame = st.session_state.get("films_fame", "No filter")
    if fame == "<500k votes":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 500_000
    elif fame == "<200k (hidden gems)":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 200_000

    mood = st.session_state.get("films_mood", "Any")
    if mood != "Any":
        prefs.mood = mood.lower()

    runtime = st.session_state.get("films_runtime", "Any")
    if runtime == "Under 100 min":
        prefs.max_runtime = 100
    elif runtime == "Under 135 min":
        prefs.max_runtime = 135

    return prefs


def _build_shows_prefs() -> ShowsPrefs:
    prefs = ShowsPrefs()
    era = st.session_state.get("shows_era", "All")
    if era == "Post-2020":
        prefs.era_min_year = 2020
    elif era == "Recent 2015+":
        prefs.era_min_year = 2015
    elif era == "Post-2000":
        prefs.era_min_year = 2000

    status = st.session_state.get("shows_status", "Any")
    if status == "Finished only":
        prefs.status_filter = "ended"
    elif status == "Ongoing only":
        prefs.status_filter = "ongoing"

    eps = st.session_state.get("shows_eps", "Any")
    if eps == "Short ≤15":
        prefs.max_episodes = 15
    elif eps == "Medium ≤50":
        prefs.max_episodes = 50

    fame = st.session_state.get("shows_fame", "No filter")
    if fame == "<200k votes":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 200_000
    elif fame == "<50k (hidden gems)":
        prefs.exclude_famous = True
        prefs.famous_votes_threshold = 50_000

    mood = st.session_state.get("shows_mood", "Any")
    if mood != "Any":
        prefs.mood = mood.lower()

    return prefs


if run_clicked:
    with st.spinner("Running pipeline…"):
        try:
            if mode_key == "films":
                prefs = _build_films_prefs()
                results = run_films(prefs)
            else:
                prefs = _build_shows_prefs()
                results = run_shows(prefs)
            st.session_state.results = results
            st.session_state.selected_title = None
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()


# ── Main content area ──────────────────────────────────────────────────────────

results = st.session_state.results
n_neighbours = results["n_neighbours"] if results else 0

tab1, tab2, tab3 = st.tabs(["Tonight's Picks", "All Candidates", "Taste Profile"])


# ── Tab 1: Tonight's Picks ─────────────────────────────────────────────────────

with tab1:
    if not results:
        st.info("Set filters in the sidebar and click **Get Recommendations**.")
    else:
        safe_bets = results.get("safe_bets", [])
        wild_cards = results.get("wild_cards", [])
        current_mode = results.get("mode", mode_key)

        filters = results.get("session_filters", "")
        if filters and filters != "none":
            st.caption(f"Active filters: {filters}")

        if safe_bets:
            st.markdown("### ✓ Safe Bets")
            for i, item in enumerate(safe_bets):
                recommendation_card(
                    item,
                    n_neighbours,
                    mode=current_mode,
                    key_prefix=f"sb_{i}",
                )
        else:
            st.warning("No safe bets found — try relaxing filters.")

        if wild_cards:
            st.markdown("### ◆ Wild Cards")
            for i, item in enumerate(wild_cards):
                recommendation_card(
                    item,
                    n_neighbours,
                    mode=current_mode,
                    key_prefix=f"wc_{i}",
                )

        recent_picks = results.get("recent_picks", [])
        if recent_picks:
            shown_titles = {x["title"] for x in safe_bets + wild_cards}
            new_recent = [x for x in recent_picks if x["title"] not in shown_titles]
            if new_recent:
                st.markdown("### ◈ Recent Picks (2020+)")
                for i, item in enumerate(new_recent):
                    recommendation_card(
                        item,
                        n_neighbours,
                        mode=current_mode,
                        key_prefix=f"rp_{i}",
                    )


# ── Tab 2: All Candidates ──────────────────────────────────────────────────────

with tab2:
    if not results or not results.get("all_scored"):
        st.info("Run the pipeline first to explore all candidates.")
    else:
        all_scored = results["all_scored"]
        current_mode = results.get("mode", mode_key)

        # Scatter plot
        fig_scatter = candidate_scatter(all_scored)
        event = st.plotly_chart(
            fig_scatter,
            use_container_width=True,
            on_select="rerun",
            key="scatter_chart",
        )

        # Handle point selection
        if event and hasattr(event, "selection") and event.selection.points:
            pt = event.selection.points[0]
            raw_cd = pt.get("customdata")
            selected_title = raw_cd[0] if isinstance(raw_cd, list) else raw_cd
            if selected_title:
                st.session_state.selected_title = selected_title

        # Inspect panel for selected point
        sel_title = st.session_state.get("selected_title")
        if sel_title:
            selected_item = next(
                (x for x in all_scored if x["title"] == sel_title), None
            )
            if selected_item:
                st.markdown(f"---\n**Inspecting: {sel_title}**")
                recommendation_card(
                    selected_item,
                    n_neighbours,
                    mode=current_mode,
                    key_prefix="inspect",
                )
                if st.button("Clear selection", key="clear_sel"):
                    st.session_state.selected_title = None
                    st.rerun()

        st.divider()

        # Sortable dataframe
        sort_col = st.selectbox(
            "Sort by",
            ["taste_score", "diversity_score", "recommended_by", "neighbour_score"],
            key="tab2_sort",
        )

        df_display = pd.DataFrame([
            {
                "Title": x["title"],
                "Year": x.get("year"),
                "Score": x.get("taste_score", 0),
                "Category": x.get("category", ""),
                "Diversity": round(x.get("diversity_score", 0), 3),
                "Neighbours": x.get("recommended_by", 0),
                "Avg★": round(x.get("neighbour_score", 0), 2),
            }
            for x in all_scored
        ])

        sort_map = {
            "taste_score": "Score",
            "diversity_score": "Diversity",
            "recommended_by": "Neighbours",
            "neighbour_score": "Avg★",
        }
        df_display = df_display.sort_values(
            sort_map.get(sort_col, "Score"), ascending=False
        ).reset_index(drop=True)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=400,
        )


# ── Tab 3: Taste Profile ───────────────────────────────────────────────────────

with tab3:
    films_tab, shows_tab = st.tabs(["Films", "Shows"])

    with films_tab:
        try:
            my_ratings_df = load_my_ratings()
            genre_stats = films_genre_stats(my_ratings_df)
            if genre_stats:
                fig_fp = taste_profile_chart(
                    genre_stats, title="My Films: Avg Rating by Genre"
                )
                st.plotly_chart(fig_fp, use_container_width=True)
                st.caption(
                    f"Based on {len(my_ratings_df)} rated films from IMDb export."
                )
            else:
                st.info("No genre data found in my_ratings.csv.")
        except FileNotFoundError:
            st.warning("my_ratings.csv not found — run `python src/ingest.py` in films/.")
        except Exception as e:
            st.error(f"Could not load films taste profile: {e}")

    with shows_tab:
        try:
            my_tv = get_my_tv_ratings()
            shows_meta = load_shows_meta()
            genre_stats_tv = shows_genre_stats(my_tv, shows_meta)
            if genre_stats_tv:
                fig_tv = taste_profile_chart(
                    genre_stats_tv, title="My Shows: Avg Rating by Genre"
                )
                st.plotly_chart(fig_tv, use_container_width=True)
                st.caption(
                    f"Based on {len(my_tv)} rated shows from MY_TV_RATINGS."
                )
            else:
                st.info("No genre data found — run `python src/ingest.py` in shows/.")
        except FileNotFoundError:
            st.warning("shows_meta.parquet not found — run `python src/ingest.py` in shows/.")
        except Exception as e:
            st.error(f"Could not load shows taste profile: {e}")
