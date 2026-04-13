"""Exploratory author-clustering visualization wrappers and registry entries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.visualizations._author_clustering_plotter import (
    plot_author_clustering,
    plot_author_reduction_comparison,
    plot_umap_parameter_comparison,
    _load_name_mapping,
)
from src.visualizations.utils import resolve_output_path

logger = logging.getLogger(__name__)
LESSON_DIR = "les6"

_CLUSTER_AGE_HINTS = {
    "Cluster 1": "jong-midden mix (ankers: Lucas 20-35, Suzanne eind 30)",
    "Cluster 2": "jongere mix (anker: Harmen eind 20)",
    "Cluster 3": "oudere groep (anker: Rene 50+)",
    "Cluster 4": "gemengde mix (ankers: Sander/Sabien/Esther)",
}

_CLUSTER_STORY = {
    "Cluster 1": "Actieve deelnemers met vergelijkbare informele schrijfstijl; ankers lopen van jongvolwassen tot eind 30.",
    "Cluster 2": "Ook een jongere mix, met een andere stijlhandtekening dan cluster 1 (leeftijd alleen verklaart het verschil dus niet).",
    "Cluster 3": "Relatief oudere ankers en een afwijkende stijlgroep; vaak compacter of anders geformuleerd taalgebruik.",
    "Cluster 4": "Gemengde leeftijden met gedeelde stijlkenmerken; suggereert dat stijl en leeftijd niet 1-op-1 samenhangen.",
}


def _les6_output_path(out_dir: str | Path | None, filename: str) -> str | Path:
    """Resolve exploratory output under the Les 6 directory."""
    base = Path(out_dir) if out_dir else Path("img")
    return resolve_output_path(base / LESSON_DIR, filename)


def author_clustering(df, out_dir: str | Path | None = None) -> None:
    """Generate exploratory author stylometric clustering scatter (t-SNE).

    :param df: Processed chat dataframe with stylometry attributes.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_author_clustering(
        df,
        out_path=_les6_output_path(out_dir, "exploratory_author_style_projection_tsne.png"),
        method="tSNE",
    )
    _export_cluster_table(df, out_dir)


def author_clustering_pca(df, out_dir: str | Path | None = None) -> None:
    """Generate exploratory author stylometric clustering scatter (PCA).

    :param df: Processed chat dataframe with stylometry attributes.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_author_clustering(
        df,
        out_path=_les6_output_path(out_dir, "exploratory_author_style_projection_pca.png"),
        method="PCA",
    )


def author_clustering_umap(df, out_dir: str | Path | None = None) -> None:
    """Generate exploratory author stylometric clustering scatter (UMAP).

    :param df: Processed chat dataframe with stylometry attributes.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_author_clustering(
        df,
        out_path=_les6_output_path(out_dir, "exploratory_author_style_projection_umap.png"),
        method="UMAP",
    )


def author_clustering_comparison(df, out_dir: str | Path | None = None) -> None:
    """Generate exploratory comparison plot for t-SNE, UMAP, and PCA.

    :param df: Processed chat dataframe with stylometry attributes.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_author_reduction_comparison(
        df,
        out_path=_les6_output_path(out_dir, "exploratory_author_style_comparison.png"),
    )


def umap_parameter_comparison(df, out_dir: str | Path | None = None) -> None:
    """Generate a side-by-side UMAP parameter comparison with cluster labels.

    Compares two UMAP configurations (n_neighbors=5/min_dist=0.1 vs
    n_neighbors=15/min_dist=0.5) so the effect of hyperparameters on the
    embedding is immediately visible.

    :param df: Processed chat dataframe with stylometry attributes.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    plot_umap_parameter_comparison(
        df,
        out_path=_les6_output_path(out_dir, "umap_parameter_comparison.png"),
    )


def _export_cluster_table(df, out_dir) -> None:
    """Write a CSV with real name, pseudo name, cluster, and message count.

    :param df: Processed chat dataframe with clustering attributes.
    :type df: Any
    :param out_dir: Optional output directory.
    :type out_dir: Any
    :return: None.
    :rtype: None
    """
    author_cluster: dict | None = df.attrs.get("stylometry_author_cluster")
    n_clusters: int = df.attrs.get("stylometry_n_clusters", 4)
    if not author_cluster:
        return

    name_map = _load_name_mapping()
    sender_counts = df["sender"].value_counts().to_dict()

    rows = []
    for pseudo, cluster_id in sorted(author_cluster.items(), key=lambda x: x[1]):
        real = name_map.get(pseudo, pseudo)
        count = sender_counts.get(pseudo, 0)
        rows.append({
            "cluster": f"Cluster {cluster_id + 1}",
            "leeftijdsindicatie": _CLUSTER_AGE_HINTS.get(f"Cluster {cluster_id + 1}", "onbekend"),
            "echte_naam": real,
            "pseudo_naam": pseudo,
            "berichten": count,
        })

    export_df = pd.DataFrame(rows).sort_values(["cluster", "berichten"], ascending=[True, False])
    out_path = _les6_output_path(out_dir, "exploratory_author_clusters.csv")
    export_df.to_csv(out_path, index=False)
    logger.info("Cluster table saved to %s", out_path)
    _export_cluster_story(export_df, out_dir)
    _export_cluster_delivery(export_df, out_dir)


def _export_cluster_story(cluster_df: pd.DataFrame, out_dir: str | Path | None = None) -> None:
    """Write a compact narrative guide for interpreting cluster results.

    :param cluster_df: Cluster summary dataframe.
    :type cluster_df: pd.DataFrame
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    out_path = Path(_les6_output_path(out_dir, "exploratory_author_cluster_story.md"))

    lines: list[str] = []
    lines.append("# Leeswijzer: Schrijfstijl-clusters")
    lines.append("")
    lines.append("Deze clusters zijn gebaseerd op schrijfstijl (trigrammen + Manhattan-afstand), niet direct op leeftijd.")
    lines.append("Leeftijd gebruik je hier als **anker**, niet als harde verklaring.")
    lines.append("")

    for cluster_name in sorted(cluster_df["cluster"].unique()):
        subset = cluster_df[cluster_df["cluster"] == cluster_name].copy()
        subset = subset.sort_values("berichten", ascending=False)
        age_hint = str(subset["leeftijdsindicatie"].iloc[0])
        top_names = ", ".join(subset["echte_naam"].head(4).tolist())
        total_msgs = int(subset["berichten"].sum())
        members = int(len(subset))
        story = _CLUSTER_STORY.get(cluster_name, "Geen extra duiding beschikbaar.")

        lines.append(f"## {cluster_name}")
        lines.append(f"- Leeftijdsanker: {age_hint}")
        lines.append(f"- Omvang: {members} personen, {total_msgs} berichten")
        lines.append(f"- Meest zichtbare namen: {top_names}")
        lines.append(f"- Interpretatie: {story}")
        lines.append("")

    lines.append("## Kort verhaal")
    lines.append("De data suggereert vier stijlgroepen. Cluster 1 (jong-midden mix) en 2 (jongere mix) liggen qua leeftijd deels bij elkaar, maar schrijven verschillend.")
    lines.append("Cluster 3 heeft een ouder anker en lijkt stijlmatig af te wijken. Cluster 4 is gemengd en ondersteunt dat stijl ≠ leeftijd.")
    lines.append("")
    lines.append("## Let op")
    lines.append("- Dit is een stijlindeling, geen persoonlijkheidsprofiel.")
    lines.append("- Leeftijdslabels zijn indicatief en gebaseerd op beperkte ankers.")
    lines.append("- Lage afstand betekent stijlgelijkenis, niet inhoudelijke overeenstemming.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Cluster story saved to %s", out_path)


def _export_cluster_delivery(cluster_df: pd.DataFrame, out_dir: str | Path | None = None) -> None:
    """Write a presentation-ready narrative with conclusions and talk track.

    :param cluster_df: Cluster summary dataframe.
    :type cluster_df: pd.DataFrame
    :param out_dir: Optional output directory.
    :type out_dir: str | Path | None
    :return: None.
    :rtype: None
    """
    out_path = Path(_les6_output_path(out_dir, "exploratory_author_cluster_delivery.md"))

    cluster_stats = (
        cluster_df.groupby("cluster")
        .agg(personen=("echte_naam", "count"), berichten=("berichten", "sum"))
        .reset_index()
        .sort_values("cluster")
    )
    total_personen = int(cluster_stats["personen"].sum())
    total_berichten = int(cluster_stats["berichten"].sum())

    lines: list[str] = []
    lines.append("# Opleververhaal: Schrijfstijl-clusters")
    lines.append("")
    lines.append("## 1. Hoofdboodschap")
    lines.append(
        "In deze chat zien we vier **schrijfstijl-clusters**. Leeftijd helpt als anker, "
        "maar verklaart de clusters niet volledig: vooral cluster 1 en 2 liggen qua leeftijd deels dicht bij elkaar, "
        "terwijl ze stijlmatig toch verschillend zijn."
    )
    lines.append("")

    lines.append("## 2. Wat is geanalyseerd")
    lines.append(f"- Dataset: {total_personen} personen, {total_berichten} berichten")
    lines.append("- Methode: karaktertrigrammen per tekstchunk, Manhattan-afstand, daarna clustering")
    lines.append("- Visualisatie: punten dicht bij elkaar = vergelijkbare schrijfstijl")
    lines.append("")

    lines.append("## 3. Uitkomsten per cluster")
    for cluster_name in cluster_stats["cluster"].tolist():
        subset = cluster_df[cluster_df["cluster"] == cluster_name].sort_values("berichten", ascending=False)
        age_hint = str(subset["leeftijdsindicatie"].iloc[0])
        top_names = ", ".join(subset["echte_naam"].head(4).tolist())
        personen = int(subset.shape[0])
        berichten = int(subset["berichten"].sum())
        lines.append(f"### {cluster_name}")
        lines.append(f"- Leeftijdsanker: {age_hint}")
        lines.append(f"- Omvang: {personen} personen, {berichten} berichten")
        lines.append(f"- Representatieve namen: {top_names}")
        lines.append(f"- Duiding: {_CLUSTER_STORY.get(cluster_name, 'Geen duiding beschikbaar.')}")
        lines.append("")

    lines.append("## 4. Wat betekent dit inhoudelijk")
    lines.append("- Schrijfstijl in deze groep lijkt door meerdere factoren te worden bepaald (niet alleen leeftijd).")
    lines.append("- Cluster 3 heeft het sterkste oudere anker en lijkt stijlmatig het meest af te wijken.")
    lines.append("- Cluster 4 is bewust als gemengde mix benoemd: dat voorkomt schijnprecisie.")
    lines.append("")

    lines.append("## 5. Grenzen van de analyse")
    lines.append("- Dit is patroonherkenning in taalvorm, geen oordeel over inhoud of persoon.")
    lines.append("- Leeftijdslabels zijn indicatief en gebaseerd op beperkte bekende ankers.")
    lines.append("- Clusters kunnen verschuiven bij andere chunkgrootte of andere afstandsmaat.")
    lines.append("")

    lines.append("## 6. Praattekst (45-60 sec)")
    lines.append(
        "\"Deze grafiek groepeert mensen op schrijfstijl, niet op mening of persoonlijkheid. "
        "We zien vier mogelijke stijlgroepen. Wat opvalt is dat cluster 1 en 2 qua leeftijd deels overlappen, "
        "maar toch stijlmatig gescheiden zijn. Dat betekent dat leeftijd wel context geeft, maar het verschil niet volledig verklaart. "
        "Cluster 3 heeft een ouder anker en lijkt taalmatig anders. Cluster 4 is een gemengde groep, "
        "wat juist laat zien dat schrijfstijl en leeftijd niet één-op-één samenhangen.\""
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Cluster delivery story saved to %s", out_path)


REGISTRY = {
    "author_clustering": author_clustering,
    "author_clustering_pca": author_clustering_pca,
    "author_clustering_umap": author_clustering_umap,
    "author_clustering_comparison": author_clustering_comparison,
    "umap_parameter_comparison": umap_parameter_comparison,
}
