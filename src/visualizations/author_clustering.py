"""Public author-clustering visualization wrappers and registry entries."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.visualizations._author_clustering_plotter import (
    plot_author_clustering,
    plot_author_reduction_comparison,
    _load_name_mapping,
)
from src.visualizations.utils import resolve_output_path

logger = logging.getLogger(__name__)

_CLUSTER_AGE_HINTS = {
    "Cluster 1": "jong-midden mix (ankers: Lucas 20-35, Suzanne eind 30)",
    "Cluster 2": "jongere mix (anker: Harmen eind 20)",
    "Cluster 3": "oudere groep (anker: Rene 50+)",
    "Cluster 4": "gemengde mix (ankers: Sander/Sabien/Esther)",
}

_CLUSTER_STORY = {
    "Cluster 1": "Veel actieve deelnemers met vergelijkbare informele schrijfstijl; ankers lopen van jongvolwassen tot eind 30.",
    "Cluster 2": "Ook een jongere mix, maar met een andere stijlhandtekening dan cluster 1 (dus leeftijd alleen verklaart het verschil niet).",
    "Cluster 3": "Relatief oudere ankers en een duidelijk andere stijlgroep; vaak compacter of anders geformuleerd taalgebruik.",
    "Cluster 4": "Gemengde leeftijden met gedeelde stijlkenmerken; dit cluster laat zien dat stijl en leeftijd niet 1-op-1 samenhangen.",
}


def author_clustering(df, out_dir: str | Path | None = None) -> None:
    """Generate the author stylometric clustering scatter plot (tSNE)."""
    plot_author_clustering(
        df,
        out_path=resolve_output_path(out_dir, "author_clustering.png"),
        method="tSNE",
    )
    _export_cluster_table(df, out_dir)


def author_clustering_pca(df, out_dir: str | Path | None = None) -> None:
    """Generate the author stylometric clustering scatter plot (PCA)."""
    plot_author_clustering(
        df,
        out_path=resolve_output_path(out_dir, "author_clustering_pca.png"),
        method="PCA",
    )


def author_clustering_umap(df, out_dir: str | Path | None = None) -> None:
    """Generate the author stylometric clustering scatter plot (UMAP)."""
    plot_author_clustering(
        df,
        out_path=resolve_output_path(out_dir, "author_clustering_umap.png"),
        method="UMAP",
    )


def author_clustering_comparison(df, out_dir: str | Path | None = None) -> None:
    """Generate compact comparison plot for t-SNE, UMAP, and PCA."""
    plot_author_reduction_comparison(
        df,
        out_path=resolve_output_path(out_dir, "author_clustering_comparison.png"),
    )


def _export_cluster_table(df, out_dir) -> None:
    """Write a CSV with real name, pseudo name, cluster, and message count."""
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
    out_path = resolve_output_path(out_dir, "author_clusters.csv")
    export_df.to_csv(out_path, index=False)
    logger.info("Cluster table saved to %s", out_path)
    _export_cluster_story(export_df, out_dir)
    _export_cluster_delivery(export_df, out_dir)


def _export_cluster_story(cluster_df: pd.DataFrame, out_dir: str | Path | None = None) -> None:
    """Write a compact narrative guide for interpreting cluster results."""
    out_path = Path(resolve_output_path(out_dir, "author_cluster_story.md"))

    lines: list[str] = []
    lines.append("# Leeswijzer: Schrijfstijl-clusters")
    lines.append("")
    lines.append("Deze clusters zijn gebaseerd op schrijfstijl (trigrammen + cosine-afstand), niet direct op leeftijd.")
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
    lines.append("De data laat vier stijlgroepen zien. Cluster 1 (jong-midden mix) en 2 (jongere mix) liggen qua leeftijd deels bij elkaar, maar schrijven verschillend.")
    lines.append("Cluster 3 heeft een ouder anker en wijkt stijlmatig af. Cluster 4 is gemengd en bevestigt dat stijl ≠ leeftijd.")
    lines.append("")
    lines.append("## Let op")
    lines.append("- Dit is een stijlindeling, geen persoonlijkheidsprofiel.")
    lines.append("- Leeftijdslabels zijn indicatief en gebaseerd op beperkte ankers.")
    lines.append("- Lage afstand betekent stijlgelijkenis, niet inhoudelijke overeenstemming.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Cluster story saved to %s", out_path)


def _export_cluster_delivery(cluster_df: pd.DataFrame, out_dir: str | Path | None = None) -> None:
    """Write a presentation-ready narrative with conclusions and talk track."""
    out_path = Path(resolve_output_path(out_dir, "author_cluster_delivery.md"))

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
    lines.append("- Methode: karaktertrigrammen per tekstchunk, cosine-afstand, daarna clustering")
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
    lines.append("- Cluster 3 heeft het duidelijkste oudere anker en wijkt stijlmatig het meest af.")
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
        "We zien vier duidelijke stijlgroepen. Wat opvalt is dat cluster 1 en 2 qua leeftijd deels overlappen, "
        "maar toch stijlmatig gescheiden zijn. Dat betekent dat leeftijd wel context geeft, maar het verschil niet volledig verklaart. "
        "Cluster 3 heeft een ouder anker en is taalmatig duidelijk anders. Cluster 4 is een gemengde groep, "
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
}
