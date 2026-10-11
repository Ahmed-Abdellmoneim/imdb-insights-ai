"""Plotly helpers that apply one visual system to every chart.

Colors come from a validated palette (light and dark steps of the same hues). Every chart
here shows a single series, so all marks use slot 1; nothing is colored by rank.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LIGHT = {
    "series": "#2a78d6",
    "text": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
DARK = {
    "series": "#3987e5",
    "text": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "surface": "#1a1a19",
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def colors() -> dict[str, str]:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return DARK if theme == "dark" else LIGHT


def _layout(fig: go.Figure, height: int, x_title: str | None, y_title: str | None) -> go.Figure:
    c = colors()
    axis = dict(
        gridcolor=c["grid"],
        linecolor=c["axis"],
        zeroline=False,
        automargin=True,  # grow the margin to fit tick labels instead of clipping them
        title_standoff=12,
        tickfont=dict(color=c["muted"], size=12),
        title_font=dict(color=c["text"], size=13),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=c["text"]),
        showlegend=False,
        hoverlabel=dict(bgcolor=c["surface"], bordercolor=c["axis"], font=dict(color=c["text"])),
        xaxis=axis | dict(title=x_title),
        yaxis=axis | dict(title=y_title),
        bargap=0.35,
    )
    return fig


def hbar(
    df: pd.DataFrame, value: str, label: str, *, value_title: str, hover: str, height: int = 420
) -> go.Figure:
    """Ranked horizontal bars, largest at the top."""
    c = colors()
    fig = go.Figure(
        go.Bar(
            x=df[value],
            y=df[label],
            orientation="h",
            marker=dict(color=c["series"], cornerradius=4),
            customdata=df,
            hovertemplate=hover + "<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    return _layout(fig, height, value_title, None)


def vbar(
    df: pd.DataFrame, x: str, y: str, *, x_title: str, y_title: str, hover: str, height: int = 340
) -> go.Figure:
    c = colors()
    fig = go.Figure(
        go.Bar(
            x=df[x],
            y=df[y],
            marker=dict(color=c["series"], cornerradius=4),
            customdata=df,
            hovertemplate=hover + "<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False)
    return _layout(fig, height, x_title, y_title)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_title: str,
    y_title: str,
    hover: str,
    diagonal: bool = False,
    log: bool = False,
    height: int = 420,
) -> go.Figure:
    """Single-series scatter; ``diagonal`` adds a muted y = x reference line."""
    c = colors()
    fig = go.Figure()
    if diagonal:
        lo = float(min(df[x].min(), df[y].min()))
        hi = float(max(df[x].max(), df[y].max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color=c["muted"], width=1),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker=dict(
                color=c["series"], size=8, opacity=0.75, line=dict(color=c["surface"], width=2)
            ),
            customdata=df,
            hovertemplate=hover + "<extra></extra>",
        )
    )
    if log:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.update_layout(hovermode="closest", hoverdistance=24)
    return _layout(fig, height, x_title, y_title)


def show(fig: go.Figure, data: pd.DataFrame, *, key: str) -> None:
    """Render a chart with its table-view twin, so no value is hover-only."""
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config={"displayModeBar": False})
    with st.expander("View data"):
        st.dataframe(data, hide_index=True, width="stretch")
