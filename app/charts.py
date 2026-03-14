"""
Plotly figure factories.
"""

from __future__ import annotations

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from app.parse_reasons import parse_all

# Category colors
COLORS = {
    "safe_bet":        "#F59E0B",  # amber
    "wild_card":       "#14B8A6",  # teal
    "skip":            "#6B7280",  # gray
    "below_threshold": "#6B7280",
}
BAR_POS = "#F59E0B"   # amber for positive contributions
BAR_NEG = "#EF4444"   # red for negative contributions


def waterfall_chart(reasons: list[str]) -> go.Figure:
    """Horizontal bar chart of score components.

    Amber bars for positive contributions, red for negative.
    """
    parsed = parse_all(reasons)
    if not parsed:
        return go.Figure()

    labels = [p[0] for p in parsed]
    values = [p[1] for p in parsed]
    bar_colors = [BAR_POS if v > 0 else BAR_NEG for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+d}" for v in values],
        textposition="auto",
        hovertemplate="%{y}: %{x:+d}<extra></extra>",
    ))

    height = max(180, len(parsed) * 34 + 50)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#374151",
            zeroline=True,
            zerolinecolor="#9CA3AF",
            zerolinewidth=1.5,
        ),
        yaxis=dict(showgrid=False, autorange="reversed"),
        showlegend=False,
        font=dict(color="#D1D5DB", size=12),
    )
    return fig


def candidate_scatter(all_scored: list[dict]) -> go.Figure:
    """Scatter of diversity_score (x) vs taste_score (y), colored by category.

    Click a point → title is returned via customdata for inspect panel.
    """
    if not all_scored:
        return go.Figure()

    df = pd.DataFrame(all_scored)
    # Normalize 'below_threshold' → 'skip' for display
    df["cat"] = df["category"].replace({"below_threshold": "skip"})

    fig = go.Figure()

    cat_order = [("safe_bet", "Safe Bet"), ("wild_card", "Wild Card"), ("skip", "Skip")]
    for cat_key, cat_label in cat_order:
        subset = df[df["cat"] == cat_key]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset.get("diversity_score", subset.get("weighted_score", [0] * len(subset))),
            y=subset["taste_score"],
            mode="markers",
            name=cat_label,
            marker=dict(
                color=COLORS.get(cat_key, "#6B7280"),
                size=9,
                opacity=0.85,
                line=dict(width=0.5, color="#1F2937"),
            ),
            customdata=subset["title"].tolist(),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Diversity: %{x:.2f}<br>"
                "Taste: %{y:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.4)",
        xaxis=dict(
            title="Diversity Score (collaborative breadth)",
            showgrid=True,
            gridcolor="#374151",
        ),
        yaxis=dict(
            title="Taste Score (personal fit)",
            showgrid=True,
            gridcolor="#374151",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(color="#D1D5DB"),
        clickmode="event+select",
    )
    return fig


def taste_profile_chart(genre_stats: dict[str, float], title: str = "") -> go.Figure:
    """Horizontal bar chart of avg rating per genre.

    Color scale: red (low) → yellow → green (high), calibrated to 5–10 range.
    """
    if not genre_stats:
        return go.Figure()

    sorted_items = sorted(genre_stats.items(), key=lambda x: -x[1])
    genres = [g for g, _ in sorted_items]
    ratings = [r for _, r in sorted_items]

    # Map 5–10 → 0–1 for colorscale; clamp outside range
    norm = [max(0.0, min(1.0, (r - 5) / 5)) for r in ratings]
    colors = pc.sample_colorscale("RdYlGn", norm)

    fig = go.Figure(go.Bar(
        x=ratings,
        y=genres,
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1f}" for r in ratings],
        textposition="auto",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))

    height = max(250, len(genres) * 28 + 60)
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 10.5],
            showgrid=True,
            gridcolor="#374151",
            title="Avg Your Rating",
        ),
        yaxis=dict(showgrid=False, autorange="reversed"),
        showlegend=False,
        font=dict(color="#D1D5DB", size=12),
    )
    return fig
