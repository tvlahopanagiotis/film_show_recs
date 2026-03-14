"""
Streamlit UI blocks — cards, pills, badges.
"""

from __future__ import annotations

import streamlit as st

from app.charts import waterfall_chart
from app.data_loader import parse_genres, strip_year

_CATEGORY_BADGE = {
    "safe_bet":        ("SAFE BET",  "#F59E0B"),
    "wild_card":       ("WILD CARD", "#14B8A6"),
    "skip":            ("SKIP",      "#6B7280"),
    "below_threshold": ("SKIP",      "#6B7280"),
}


def _fmt_year(year) -> str:
    if year is None:
        return ""
    try:
        return str(int(year))
    except (ValueError, TypeError):
        return ""


def _badge_md(category: str) -> str:
    label, color = _CATEGORY_BADGE.get(category, ("?", "#6B7280"))
    return (
        f'<span style="background:{color};color:#000;'
        f'font-weight:700;padding:2px 8px;border-radius:4px;'
        f'font-size:0.75rem;letter-spacing:0.05em;">{label}</span>'
    )


def recommendation_card(
    item: dict,
    n_neighbours: int,
    mode: str = "films",
    key_prefix: str = "",
) -> None:
    """Render a recommendation card with inline top-3 reasons and waterfall expander."""
    title_raw = item.get("title", "Unknown")
    title = strip_year(title_raw) if mode == "films" else title_raw
    year = _fmt_year(item.get("year"))

    genres = item.get("genres", [])
    if isinstance(genres, str):
        genres = parse_genres(genres)
    if mode == "shows":
        genre_str = " · ".join(g.title() for g in genres[:4])
    else:
        genre_str = " · ".join(g for g in genres[:4])

    taste_score = item.get("taste_score", 0)
    recommended_by = item.get("recommended_by", 0)
    neighbour_score = item.get("neighbour_score", 0)
    category = item.get("category", "")
    reasons = item.get("reasons", [])

    # Extra shows metadata
    status = item.get("status", "")
    aired_eps = item.get("aired_episodes")

    with st.container(border=True):
        # Header row: badge + score
        head_col, score_col = st.columns([4, 1])
        with head_col:
            year_str = f" ({year})" if year else ""
            badge = _badge_md(category)
            st.markdown(
                f"{badge} &nbsp; **{title}{year_str}**",
                unsafe_allow_html=True,
            )
        with score_col:
            st.markdown(
                f'<div style="text-align:right;font-size:1.3rem;'
                f'font-weight:700;color:#F59E0B;">{taste_score:.0f}</div>',
                unsafe_allow_html=True,
            )

        # Genre line
        if genre_str:
            st.caption(genre_str)

        # Shows status line
        if mode == "shows" and status:
            status_map = {
                "ended": "Ended", "canceled": "Cancelled",
                "returning series": "Ongoing", "in production": "In Production",
            }
            status_str = status_map.get(status.lower(), status)
            eps_str = f" · {int(aired_eps)} ep" if aired_eps else ""
            st.caption(f"{status_str}{eps_str}")

        # Neighbour stats
        st.caption(
            f"**{recommended_by}/{n_neighbours}** neighbours &nbsp;·&nbsp; avg {neighbour_score:.1f}★"
        )

        # Top 3 reasons inline
        for r in reasons[:3]:
            st.markdown(f"&nbsp;&nbsp;&nbsp;↳ {r}", unsafe_allow_html=True)

        # Expandable full waterfall
        if reasons:
            safe_key = f"{key_prefix}_{title_raw[:25].replace(' ', '_')}"
            with st.expander("Full score breakdown"):
                fig = waterfall_chart(reasons)
                st.plotly_chart(fig, use_container_width=True, key=f"wf_{safe_key}")
