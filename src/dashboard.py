"""Streamlit dashboard for interactive DAV chat analysis and visualization generation."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
import streamlit as st

# Ensure `src` imports resolve when running: `streamlit run src/dashboard.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.visualizations.registry import load_registry, run_selected

NAVY = "#0B2545"
NAVY_MID = "#1E3A5F"
NAVY_LIGHT = "#2F5B8A"
SKY = "#7FA6D6"
SURFACE = "#F4F8FF"
GRID = "#D9E4F2"
INCIDENT = "#C62828"


def _load_data(csv_path: Path) -> pd.DataFrame:
    """Load processed CSV and normalize key types for dashboard usage."""
    if not csv_path.exists():
        st.error(
            "Processed data not found. Run `uv run python -m src.main` first to generate files."
        )
        st.stop()

    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date_only"] = pd.to_datetime(df["date_only"], errors="coerce").dt.date
    if "has_emoji" not in df.columns and "contains_emoji" in df.columns:
        df["has_emoji"] = df["contains_emoji"].fillna(False).astype(bool)
    if "contains_emoji" not in df.columns and "has_emoji" in df.columns:
        df["contains_emoji"] = df["has_emoji"].fillna(False).astype(bool)
    if "emoji_list" in df.columns:
        df["emoji_list"] = df["emoji_list"].apply(_parse_emoji_list)
    # emoji_group is not saved to CSV — compute it on the fly
    if "emoji_group" not in df.columns and "emoji_list" in df.columns:
        try:
            from src.modules.feature_engineering import add_emoji_category
            df = add_emoji_category(df)
        except (KeyError, ValueError) as exc:
            logger.warning("Could not add emoji_group column: %s", exc)
    return df


def _parse_emoji_list(value: object) -> list[str]:
    """Parse serialized emoji-list values into Python lists."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return []


def _render_kpis(df: pd.DataFrame) -> None:
    """Show high-level dashboard KPI cards."""
    total_messages = len(df)
    unique_senders = int(df["sender"].nunique())
    incident_messages = int(df["is_incident_message"].fillna(0).sum())
    emoji_messages = int(df["has_emoji"].fillna(False).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Berichten", f"{total_messages:,}".replace(",", "."))
    col2.metric("Auteurs", unique_senders)
    col3.metric("Incident-berichten", f"{incident_messages:,}".replace(",", "."))
    col4.metric("Emoji-berichten", f"{emoji_messages:,}".replace(",", "."))


def _apply_dashboard_theme() -> None:
    """Inject a navy dashboard theme for consistent figure-ground hierarchy."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
        }}
        .block-container {{
            padding-top: 1.75rem;
        }}
        [data-testid="stSidebar"] {{
            display: none;
        }}
        [data-testid="collapsedControl"] {{
            display: none;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stDeployButton"],
        [data-testid="stAppDeployButton"],
        [data-testid*="deploy" i],
        button[title="Deploy"],
        button[aria-label="Deploy"],
        a[title="Deploy"] {{
            display: none;
        }}
        [data-testid="stToolbar"],
        [data-testid="stHeaderActionElements"] {{
            top: 0.45rem;
            right: 0.9rem;
            background: transparent;
            border: 0;
            box-shadow: none;
        }}
        [data-testid="stToolbar"] > div,
        [data-testid="stHeaderActionElements"] > div {{
            background: transparent;
            border: 0;
            box-shadow: none;
            padding: 0;
        }}
        [data-testid="stToolbar"] button,
        [data-testid="stHeaderActionElements"] button {{
            background: transparent;
            border: 0;
            box-shadow: none;
            color: {NAVY};
        }}
        [data-testid="stMetricValue"] {{
            color: {NAVY};
        }}
        h1, h2, h3 {{
            color: {NAVY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_figure(fig: go.Figure) -> go.Figure:
    """Apply shared navy styling so all visuals read as one system."""
    fig.update_layout(
        template="simple_white",
        font={"color": NAVY},
        title={"font": {"color": NAVY, "size": 20}},
        paper_bgcolor="white",
        plot_bgcolor=SURFACE,
        colorway=[NAVY_LIGHT, SKY, INCIDENT, NAVY_MID],
        hoverlabel={"bgcolor": "white", "font_color": NAVY, "bordercolor": NAVY_LIGHT},
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def _resampled_message_counts(df: pd.DataFrame, freq: str) -> pd.Series:
    """Aggregate message counts on a datetime frequency."""
    return (
        df.dropna(subset=["datetime"])
        .set_index("datetime")
        .resample(freq)
        .size()
        .rename("count")
    )


def _hourly_message_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate message counts by hour of day."""
    return (
        df.dropna(subset=["hour"])
        .groupby("hour", as_index=False)
        .size()
        .rename(columns={"size": "messages"})
        .sort_values("hour")
    )


def _apply_filters(
    df: pd.DataFrame,
    selected_dates: tuple | list | object,
) -> pd.DataFrame:
    """Apply date filters from the sidebar."""
    filtered = df.copy()
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered = filtered[
            (filtered["date_only"] >= start_date) & (filtered["date_only"] <= end_date)
        ]
    return filtered


def _plot_overall_emoji_distribution(df: pd.DataFrame) -> go.Figure:
    counts = (
        df[df["emoji_group"].notna()]
        .groupby("emoji_group", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    counts["proportion"] = counts["count"] / max(float(counts["count"].sum()), 1.0)
    fig = px.bar(
        counts,
        x="emoji_group",
        y="proportion",
        color="emoji_group",
        color_discrete_sequence=[NAVY_LIGHT, SKY, NAVY_MID, "#4B6B93"],
        text=counts["proportion"].map(lambda value: f"{value:.1%}"),
        hover_data={"count": True, "proportion": ":.2%"},
        title="Emoji-distributie per categorie",
        labels={"emoji_group": "Emoji-categorie", "proportion": "Aandeel"},
    )
    return _style_figure(fig)


def _plot_chat_activity_by_hour(df: pd.DataFrame) -> go.Figure:
    hourly = _hourly_message_counts(df)
    fig = px.bar(
        hourly,
        x="hour",
        y="messages",
        color_discrete_sequence=[NAVY_LIGHT],
        hover_data={"hour": True, "messages": True},
        title="Chatactiviteit per uur",
        labels={"hour": "Uur", "messages": "Aantal berichten"},
    )
    return _style_figure(fig)


def _plot_chat_activity_weekday_weekend(df: pd.DataFrame) -> go.Figure:
    working = df.dropna(subset=["datetime"]).copy()
    working["date_only"] = working["datetime"].dt.date
    working["group"] = np.where(working["datetime"].dt.weekday >= 5, "Weekend", "Weekdag")
    daily = working.groupby(["date_only", "group"], as_index=False).size().rename(columns={"size": "messages"})
    fig = px.strip(
        daily,
        x="group",
        y="messages",
        color="group",
        color_discrete_map={"Weekdag": NAVY_MID, "Weekend": INCIDENT},
        hover_data={"date_only": True, "messages": True},
        title="Weekdag vs weekend: berichten per dag",
        labels={"group": "Groep", "messages": "Berichten per dag"},
    )
    fig.update_traces(jitter=0.25, marker={"size": 8, "opacity": 0.7})
    return _style_figure(fig)


def _plot_emoji_usage_by_hour(df: pd.DataFrame) -> go.Figure:
    hourly = (
        df.groupby("hour", as_index=False)["has_emoji"]
        .mean()
        .rename(columns={"has_emoji": "emoji_probability"})
        .sort_values("hour")
    )
    fig = px.bar(
        hourly,
        x="hour",
        y="emoji_probability",
        color_discrete_sequence=[NAVY_MID],
        hover_data={"emoji_probability": ":.2%"},
        title="Kans op emoji per uur",
        labels={"hour": "Uur", "emoji_probability": "Emoji-kans"},
    )
    fig.add_scatter(
        x=hourly["hour"],
        y=hourly["emoji_probability"].rolling(window=3, center=True, min_periods=1).mean(),
        mode="lines",
        name="3-uurs trend",
    )
    return _style_figure(fig)


def _weekly_incident_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.dropna(subset=["datetime"]).copy()
    working["date_only"] = pd.to_datetime(working["date_only"], errors="coerce")
    daily = (
        working.groupby("date_only", as_index=False)
        .agg(
            total_message_count=("datetime", "size"),
            incident_message_count=("is_incident_message", "sum"),
        )
        .dropna(subset=["date_only"])
    )
    weekly = (
        daily.set_index("date_only")
        .resample("W-MON")
        .sum(numeric_only=True)
        .reset_index()
        .rename(columns={"date_only": "week_start"})
    )
    weekly["incident_ratio_pct"] = (
        weekly["incident_message_count"] / weekly["total_message_count"].replace(0, pd.NA) * 100
    ).fillna(0.0)
    return weekly


def _plot_incident_activity_correlation(df: pd.DataFrame) -> go.Figure:
    weekly = _weekly_incident_frame(df)
    fig = px.scatter(
        weekly,
        x="total_message_count",
        y="incident_message_count",
        trendline="lowess",
        color_discrete_sequence=[NAVY_LIGHT],
        hover_data={"week_start": True, "incident_ratio_pct": ":.2f"},
        title="Relatie: totaal activiteit vs incident-activiteit (week)",
        labels={
            "total_message_count": "Totaal berichten per week",
            "incident_message_count": "Incident-berichten per week",
        },
    )
    return _style_figure(fig)


def _plot_incident_discussion_timeline(df: pd.DataFrame) -> go.Figure:
    weekly = _weekly_incident_frame(df)
    fig = go.Figure()
    fig.add_bar(
        x=weekly["week_start"],
        y=weekly["total_message_count"],
        name="Totaal berichten",
        marker={"color": NAVY_LIGHT},
        hovertemplate="Week: %{x|%Y-%m-%d}<br>Totaal: %{y}<extra></extra>",
    )
    fig.add_bar(
        x=weekly["week_start"],
        y=weekly["incident_message_count"],
        name="Incident-berichten",
        marker={"color": INCIDENT},
        hovertemplate="Week: %{x|%Y-%m-%d}<br>Incident: %{y}<extra></extra>",
    )
    fig.add_scatter(
        x=weekly["week_start"],
        y=weekly["incident_ratio_pct"],
        mode="lines+markers",
        yaxis="y2",
        name="Incident-ratio (%)",
        line={"color": NAVY_MID, "width": 2},
        hovertemplate="Week: %{x|%Y-%m-%d}<br>Ratio: %{y:.2f}%<extra></extra>",
    )
    fig.update_layout(
        title="Incidentdiscussie in de tijd (wekelijks)",
        yaxis={"title": "Aantal berichten"},
        yaxis2={"title": "Incident-ratio (%)", "overlaying": "y", "side": "right"},
        barmode="overlay",
    )
    return _style_figure(fig)


def _plot_time_series_activity(df: pd.DataFrame) -> go.Figure:
    ts = _resampled_message_counts(df, "15min").reset_index()
    ts["rolling"] = ts["count"].rolling(window=8, center=True, min_periods=1).mean()
    fig = px.line(
        ts,
        x="datetime",
        y="count",
        color_discrete_sequence=[NAVY_LIGHT],
        title="Tijdreeks activiteit (15 min)",
        labels={"datetime": "Tijd", "count": "Berichten per 15 min"},
    )
    fig.add_scatter(
        x=ts["datetime"],
        y=ts["rolling"],
        mode="lines",
        name="Rollend gemiddelde (2u)",
    )
    return _style_figure(fig)


def _plot_time_series_autocorrelation(df: pd.DataFrame) -> go.Figure:
    ts = _resampled_message_counts(df, "15min")
    counts = ts.values.astype(float)
    n = len(counts)
    if n < 2:
        raise ValueError("Te weinig data voor autocorrelatie.")
    centered = counts - counts.mean()
    full = np.correlate(centered, centered, mode="full")
    acf = full[n - 1 :] / full[n - 1]
    max_lag = min(96, n - 1)
    lags = np.arange(1, max_lag + 1)
    lag_hours = lags * 15 / 60
    data = pd.DataFrame({"lag_hours": lag_hours, "acf": acf[1 : max_lag + 1]})
    fig = px.bar(
        data,
        x="lag_hours",
        y="acf",
        color_discrete_sequence=[NAVY_MID],
        title="Autocorrelatie van chatactiviteit",
        labels={"lag_hours": "Vertraging (uur)", "acf": "Autocorrelatie"},
    )
    return _style_figure(fig)


def _plot_poisson_model(df: pd.DataFrame) -> go.Figure:
    ts = _resampled_message_counts(df, "1h")
    lam = float(ts.mean())
    n_intervals = len(ts)
    max_count = int(ts.max())
    x_values = np.arange(0, max_count + 1)
    observed = np.array([(ts == count).sum() for count in x_values])
    expected = poisson.pmf(x_values, lam) * n_intervals
    fig = go.Figure()
    fig.add_bar(
        x=x_values,
        y=observed,
        name="Waargenomen",
        hovertemplate="Berichten/uur: %{x}<br>Frequentie: %{y}<extra></extra>",
    )
    fig.add_scatter(
        x=x_values,
        y=expected,
        mode="lines+markers",
        name=f"Poisson verwachting (λ={lam:.2f})",
        line={"color": INCIDENT, "width": 2.4},
        hovertemplate="Berichten/uur: %{x}<br>Verwacht: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        title="Poisson-model: waargenomen vs verwacht",
        xaxis_title="Berichten per uur",
        yaxis_title="Frequentie",
    )
    return _style_figure(fig)


def _matched_bow_terms(text: str, terms: list[str]) -> str:
    """Return comma-separated BoW terms that appear in *text*."""
    import re
    text_lower = text.lower()
    matched = [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", text_lower)]
    return ", ".join(matched) if matched else "-"


def _build_incident_sample(df: pd.DataFrame, n_max: int = 700) -> pd.DataFrame:
    """Return a balanced, shuffled sample of incident vs regular messages.

    Selects relevant columns, filters short messages, and balances classes.

    :param df: Processed chat dataframe.
    :param n_max: Maximum messages per class.
    :raises ValueError: If either class is empty after filtering.
    :return: Balanced dataframe ready for vectorization.
    """
    keep_cols = ["message", "is_incident_message"]
    if "datetime" in df.columns:
        keep_cols.append("datetime")
    if "sender" in df.columns:
        keep_cols.append("sender")

    sample = df[keep_cols].dropna(subset=["message"]).copy()
    sample["message"] = sample["message"].astype(str).str.strip()
    sample = sample[sample["message"].str.len() >= 8]
    incident = sample[sample["is_incident_message"] == 1]
    regular = sample[sample["is_incident_message"] == 0]
    if incident.empty or regular.empty:
        raise ValueError("Incident-context heeft beide klassen nodig.")
    return pd.concat(
        [
            incident.sample(n=min(len(incident), n_max), random_state=42),
            regular.sample(n=min(len(regular), n_max), random_state=42),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _plot_incident_context_projection(df: pd.DataFrame, method: str = "UMAP") -> go.Figure:
    """Project messages into 2D using TF-IDF + selected method, colored by incident label."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from src.modules.feature_engineering import INCIDENT_BOW_TERMS

    balanced = _build_incident_sample(df)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=500, min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(balanced["message"].tolist()).toarray()

    # Welke incident-BoW termen heeft dit bericht getriggerd?
    bow_matches = [
        _matched_bow_terms(msg, INCIDENT_BOW_TERMS)
        for msg in balanced["message"].tolist()
    ]

    # Dimensiereductie op basis van gekozen methode
    method_key = method.strip().upper().replace("-", "").replace("_", "")
    if method_key == "PCA":
        emb = PCA(n_components=2, random_state=42).fit_transform(X)
        title = "Incident-context projectie (PCA)"
    elif method_key == "TSNE":
        n = len(balanced)
        perplexity = min(30, max(5, n // 3))
        emb = TSNE(
            n_components=2, perplexity=perplexity, metric="euclidean",
            init="random", random_state=42, max_iter=1000,
        ).fit_transform(X)
        title = "Incident-context projectie (t-SNE)"
    else:  # UMAP
        import umap as umap_lib  # type: ignore[import-not-found]
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.10,
            random_state=42, init="random",  # random init: stabieler op sparse TF-IDF
        )
        emb = reducer.fit_transform(X)
        title = "Incident-context projectie (UMAP)"

    # Bouw plot-dataframe
    embed_df = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "incident_related": balanced["is_incident_message"].eq(1),
        "bericht": balanced["message"].str[:120].str.replace("\n", " "),
        "incident_termen": bow_matches,
    })
    if "datetime" in balanced.columns:
        embed_df["datum"] = pd.to_datetime(balanced["datetime"]).dt.strftime("%Y-%m-%d")
    if "sender" in balanced.columns:
        embed_df["auteur"] = balanced["sender"]

    # Hover kolommen
    hover_cols = {"x": False, "y": False, "incident_related": False,
                  "bericht": True, "incident_termen": True}
    if "datum" in embed_df.columns:
        hover_cols["datum"] = True
    if "auteur" in embed_df.columns:
        hover_cols["auteur"] = True

    fig = px.scatter(
        embed_df,
        x="x", y="y",
        color="incident_related",
        color_discrete_map={False: SKY, True: INCIDENT},
        title=title,
        labels={"incident_related": "Incident-gerelateerd"},
        hover_data=hover_cols,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.75})
    return _style_figure(fig)


INTERACTIVE_BUILDERS: dict[str, Callable[[pd.DataFrame], go.Figure]] = {
    "overall_emoji_distribution": _plot_overall_emoji_distribution,
    "chat_activity_by_hour": _plot_chat_activity_by_hour,
    "chat_activity_weekday_weekend": _plot_chat_activity_weekday_weekend,
    "emoji_usage_by_hour": _plot_emoji_usage_by_hour,
    "incident_activity_correlation": _plot_incident_activity_correlation,
    "incident_discussion_timeline": _plot_incident_discussion_timeline,
    "time_series_activity": _plot_time_series_activity,
    "time_series_autocorrelation": _plot_time_series_autocorrelation,
    "poisson_model": _plot_poisson_model,
}

VIS_LABELS: dict[str, str] = {
    "chat_activity_by_hour": "Activiteit per uur",
    "overall_emoji_distribution": "Emoji verdeling",
    "incident_discussion_timeline": "Incidenten in de tijd",
    "chat_activity_weekday_weekend": "Weekdag vs weekend",
    "emoji_usage_by_hour": "Emoji kans per uur",
    "incident_activity_correlation": "Activiteit vs incident correlatie",
    "time_series_activity": "Tijdreeks (15 min)",
    "time_series_autocorrelation": "Autocorrelatie",
    "poisson_model": "Poisson model",
    "dimensiereductie": "Dimensiereductie",
}

BASIC_VISUALIZATIONS = [
    "chat_activity_by_hour",
    "overall_emoji_distribution",
    "incident_discussion_timeline",
    "emoji_usage_by_hour",
    "poisson_model",
    "dimensiereductie",
]

ALL_VISUALIZATIONS = [
    "chat_activity_by_hour",
    "overall_emoji_distribution",
    "incident_discussion_timeline",
    "chat_activity_weekday_weekend",
    "emoji_usage_by_hour",
    "incident_activity_correlation",
    "time_series_activity",
    "time_series_autocorrelation",
    "poisson_model",
    "dimensiereductie",
]


def _plot_dimensiereductie_vergelijking(df: pd.DataFrame) -> go.Figure:
    """Interactieve t-SNE vs UMAP vergelijking met hover (BoW termen + datum + bericht)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from plotly.subplots import make_subplots
    from src.modules.feature_engineering import INCIDENT_BOW_TERMS

    balanced = _build_incident_sample(df)

    vectorizer = TfidfVectorizer(max_features=500, min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(balanced["message"].tolist()).toarray()

    # Gedeelde hover data
    bow_matches = [_matched_bow_terms(m, INCIDENT_BOW_TERMS) for m in balanced["message"]]
    hover_msg = balanced["message"].str[:120].str.replace("\n", " ").tolist()
    is_inc = balanced["is_incident_message"].eq(1).tolist()
    datum = pd.to_datetime(balanced["datetime"]).dt.strftime("%Y-%m-%d").tolist() if "datetime" in balanced.columns else [""] * len(balanced)
    auteur = balanced["sender"].tolist() if "sender" in balanced.columns else [""] * len(balanced)

    # t-SNE embedding
    n = len(balanced)
    perplexity = min(30, max(5, n // 3))
    emb_tsne = TSNE(n_components=2, perplexity=perplexity, metric="euclidean",
                    init="random", random_state=42, max_iter=1000).fit_transform(X)

    # UMAP embedding
    import umap as umap_lib  # type: ignore[import-not-found]
    emb_umap = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.10,
                              random_state=42, init="random").fit_transform(X)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["t-SNE · lokale structuur", "UMAP · globale + lokale structuur"])

    for col_idx, (emb, name) in enumerate([(emb_tsne, "t-SNE"), (emb_umap, "UMAP")], start=1):
        for label_val, label_name, color, symbol in [
            (False, "Regulier", SKY, "circle"),
            (True, "Incident", INCIDENT, "circle"),
        ]:
            mask = [inc == label_val for inc in is_inc]
            x_pts = emb[mask, 0].tolist()
            y_pts = emb[mask, 1].tolist()
            custom = [
                [hover_msg[i], bow_matches[i], datum[i], auteur[i]]
                for i, m in enumerate(mask) if m
            ]
            fig.add_trace(
                go.Scatter(
                    x=x_pts, y=y_pts,
                    mode="markers",
                    name=label_name,
                    legendgroup=label_name,
                    showlegend=(col_idx == 1),
                    marker=dict(color=color, size=7 if label_val else 5,
                                opacity=0.85 if label_val else 0.45,
                                line=dict(width=0.4, color="white") if label_val else dict(width=0)),
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{customdata[3]}</b> · %{customdata[2]}<br>"
                        "💬 %{customdata[0]}<br>"
                        "🔑 %{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title_text="t-SNE vs UMAP · karakter-trigrammen · rood = incident-gerelateerd",
        height=600,
        template="simple_white",
        font={"color": NAVY},
        paper_bgcolor="white",
        plot_bgcolor=SURFACE,
        hoverlabel={"bgcolor": "white", "font_color": NAVY, "bordercolor": NAVY_LIGHT},
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def main() -> None:
    """Run Streamlit dashboard app."""
    st.set_page_config(page_title="DAV Chat Dashboard", page_icon="📊", layout="wide")
    _apply_dashboard_theme()

    df = _load_data(config.PROCESSED_DIR / "clean_chat_processed.csv")

    min_date = df["date_only"].dropna().min()
    max_date = df["date_only"].dropna().max()

    # Header + datumfilter inline (Elastic-stijl)
    title_col, date_col = st.columns([3, 1])
    with title_col:
        st.title("DAV Chat Dashboard")
    with date_col:
        st.markdown("<div style='padding-top:1.6rem'></div>", unsafe_allow_html=True)
        selected_dates = st.date_input(
            "",
            value=(min_date, max_date) if min_date and max_date else None,
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )

    filtered = _apply_filters(df, selected_dates)
    _render_kpis(filtered)
    st.divider()

    # Navigatie als tabs
    tab_labels = [VIS_LABELS[k] for k in ALL_VISUALIZATIONS]
    tabs = st.tabs(tab_labels)

    for tab, key in zip(tabs, ALL_VISUALIZATIONS):
        with tab:
            try:
                if key == "dimensiereductie":
                    col_left, col_right = st.columns([3, 1])
                    with col_right:
                        weergave = st.radio(
                            "Weergave",
                            ["Vergelijking (t-SNE + UMAP)", "Enkel (kies methode)"],
                            key="dim_weergave",
                        )
                    if weergave == "Vergelijking (t-SNE + UMAP)":
                        with st.spinner("t-SNE en UMAP berekenen..."):
                            figure = _plot_dimensiereductie_vergelijking(filtered)
                        st.plotly_chart(figure, use_container_width=True)
                    else:
                        with col_right:
                            method = st.selectbox(
                                "Methode",
                                ["UMAP", "tSNE", "PCA"],
                                key="dim_methode",
                            )
                        with st.spinner("Embedding berekenen..."):
                            figure = _plot_incident_context_projection(filtered, method=method)
                        st.plotly_chart(figure, use_container_width=True)

                else:
                    figure = INTERACTIVE_BUILDERS[key](filtered)
                    st.plotly_chart(figure, use_container_width=True)
            except (KeyError, ValueError, RuntimeError) as exc:
                st.warning(f"Visualisatie kon niet worden gemaakt: {exc}")

    st.divider()
    with st.expander("Geavanceerd: pipeline-visualisaties genereren", expanded=False):
        registry = load_registry()
        all_vis = sorted(registry.keys())
        selected_to_generate = st.multiselect(
            "Selecteer visualisaties om te genereren (PNG)",
            options=all_vis,
            default=[],
        )
        use_filtered = st.checkbox("Gebruik huidige filters voor generatie", value=False)

        if st.button("Genereer geselecteerde visualisaties", type="primary"):
            generation_df = filtered if use_filtered else df
            with st.spinner("Visualisaties genereren..."):
                run_selected(
                    generation_df.copy(),
                    {name: name in selected_to_generate for name in all_vis},
                    out_dir=config.IMG_DIR,
                )
            st.success(f"Klaar. Bestanden staan in `{config.IMG_DIR}`.")


if __name__ == "__main__":
    main()
