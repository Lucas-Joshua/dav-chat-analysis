# DAV Chat Analysis

## Studentinformatie

Naam: Lucas Joshua\
Studentnummer: 1905781

## Projectbeschrijving

Dit project is ontwikkeld voor het vak **Data Analysis & Visualisation (DAV)**.

De applicatie analyseert een WhatsApp-chatexport (`.txt`) en doorloopt een volledige data-pipeline: van ruwe tekst tot geaggregeerde statistieken, feature engineering, anonimisering en interactieve visualisaties. Als verdieping bevat het project een **schrijfstijlanalyse (stylometrie)** op basis van karaktertrigrammen, waarbij K-means clustering en dimensiereductie (PCA, t-SNE, UMAP) worden ingezet om auteurs te groeperen op schrijfstijl — zonder te kijken naar wat iemand schrijft, alleen *hoe*.

---

## Pipeline-stappen

```
raw_data.txt
    │
    ├─ 1. Laden          data_loader.py       → ruwe berichten inlezen
    ├─ 2. Opschonen      data_cleaning.py     → encoding, lege regels, systeemberichten
    ├─ 3. Anonimiseren   anonymizer.py        → echte namen → pseudo-namen (user_mapping.csv)
    ├─ 4. Preprocessing  preprocessor.py      → datum/tijd parsing, auteur extractie
    ├─ 5. Features       feature_engineering  → berichtlengte, tijdfeatures, emoji,
    │                                            links, scheldwoorden (BOW), incident-labels
    ├─ 6. Stylometrie    author_stylometry.py → karaktertrigrammen → CountVectorizer →
    │                                            Manhattan-afstand → PCA / t-SNE / UMAP →
    │                                            K-means clustering (4 clusters)
    ├─ 7. Opslaan        pipeline.py          → data/processed/ (CSV + Parquet)
    └─ 8. Visualisaties  visualizations/      → img/ (PNG)
```

---

## Visualisaties

| Naam | Bestand | Beschrijving |
|------|---------|--------------|
| `author_clustering` | `author_clustering.py` | t-SNE scatter met schrijfstijl-clusters en Gestalt-principes |
| `author_clustering_pca` | idem | Zelfde visualisatie op PCA-embedding |
| `author_clustering_umap` | idem | Zelfde visualisatie op UMAP-embedding |
| `author_clustering_comparison` | idem | Zij-aan-zij vergelijking PCA / t-SNE / UMAP |
| `chat_activity_by_hour` | `chat_activity.py` | Berichtvolume per uur van de dag |
| `chat_activity_distribution` | idem | Distributie van berichtfrequentie per auteur |
| `response_time_suite` | `response_time_suite.py` | Reactietijden tussen auteurs |
| `emoji_*` | `emoji.py` | Emoji-gebruik per auteur, per uur, heatmap |
| `negative_reaction_*` | `negative_reactions.py` | Negatieve reacties (concentratie, scatter, diagnostiek) |
| `incident_*` | `incident_timeline.py` | Incident-tijdlijn en activiteitspatronen |

### Gestalt-principes in de schrijfstijl-plot

De clusterchart past vijf Gestalt-principes toe om de informatie direct leesbaar te maken:

- **Common Region** – vertrouwensellipsen (numpy eigendecompositie) omsluiten de tekstfragmenten per auteur
- **Similarity** – alle punten van één cluster krijgen dezelfde kleur
- **Figure/Ground** – grijze achtergrondpunten laten de gekleurde clusters uitspringen
- **Proximity** – namen staan direct bij hun zwaartepunt (centroid), geen aparte legenda nodig
- **Closure** – het storyboard-raster toont elk cluster apart zodat de grens direct zichtbaar is

---

## Stylometrie — hoe werkt het?

1. Berichten per auteur worden samengevoegd tot één corpus en opgedeeld in chunks van 500 tekens.
2. `CountVectorizer` zet elke chunk om naar een frequentievector van karaktertrigrammen (bv. `"hoe"`, `"oe "`, `"e j"`).
3. De vectoren worden vergeleken via **Manhattan-afstand** — een robuuste maat voor stijlverschillen.
4. **PCA**, **t-SNE** en **UMAP** reduceren de hoge-dimensionale ruimte naar 2D (of 3D) voor visualisatie.
5. **K-means** (k=4) clustert auteurs op basis van de t-SNE-embedding.

### Gevonden clusters (referentiepunten van bekende deelnemers)

| Cluster | Kleur | Leeftijdsindicatie | Basis |
|---------|-------|--------------------|-------|
| Cluster 1 | rood | ~20–35 jaar | Lucas Joshua (20–35) |
| Cluster 2 | groen | ~eind 20 | Harmen Jaarsma (eind 20) |
| Cluster 3 | oranje | ~50+ jaar | Rene Warries (~50) |
| Cluster 4 | blauw | ~35–60 jaar (breed) | Sander (~38), Sabien (~48), Esther (~58) |

> **Let op:** schrijfstijl ≠ leeftijd. Clusters 1 en 2 overlappen qua leeftijd maar hebben aantoonbaar verschillende schrijfstijlen. Leeftijd is een *indicatieve* observatie, geen causale conclusie.

---

## Projectstructuur

```
dav-chat-analysis/
├── src/
│   ├── main.py                        # Entry point
│   ├── pipeline.py                    # Pipeline-orkestrator
│   ├── config.py                      # Pad- en parameterinstellingen
│   ├── logging_config.py
│   ├── modules/
│   │   ├── data_loader.py
│   │   ├── data_cleaning.py
│   │   ├── anonymizer.py
│   │   ├── preprocessor.py
│   │   ├── feature_engineering.py
│   │   ├── metadata.py
│   │   └── author_stylometry.py       # Stylometrie + embeddings
│   └── visualizations/
│       ├── registry.py                # Visualisatie-registry
│       ├── plot_settings.py           # Gedeelde stijlinstellingen
│       ├── author_clustering.py       # Schrijfstijl-clustering
│       ├── _author_clustering_plotter.py  # Plotly low-level renderer
│       ├── chat_activity.py
│       ├── emoji.py
│       ├── negative_reactions.py
│       ├── incident_timeline.py
│       └── response_time_suite.py
├── data/
│   ├── raw/                           # Onbewerkte WhatsApp-export
│   └── processed/                     # Verwerkte data (CSV + Parquet)
│       └── user_mapping.csv           # Pseudo-naam ↔ echte naam mapping
├── img/                               # Gegenereerde visualisaties (PNG)
│   └── author_clusters.csv            # Cluster-indeling per auteur
├── notebooks/                         # Jupyter notebooks (exploratie)
├── logs/                              # Logbestanden
├── pyproject.toml
└── readme.md
```

---

## Installatie

Gebruik `uv` om de omgeving op te zetten:

```bash
uv sync
```

---

## Uitvoeren van de pipeline

```bash
uv run python -m src.main
```

Na uitvoering worden:
- Verwerkte data opgeslagen in `data/processed/`
- Visualisaties opgeslagen in `img/`
- Logs opgeslagen in `logs/`

Specifieke visualisaties aan- of uitzetten kan via de `VISUALIZATION_SELECTIONS`-dict in `src/pipeline.py`.

---

## Gebruikte technologieën

| Pakket | Gebruik |
|--------|---------|
| `pandas` | Data-manipulatie en pipeline |
| `numpy` | Matrices, eigendecompositie (ellipsen) |
| `scikit-learn` | CountVectorizer, PCA, t-SNE, K-means |
| `umap-learn` | UMAP-dimensiereductie |
| `plotly` + `kaleido` | Interactieve plots → PNG export |
| `matplotlib` / `seaborn` | Statische vergelijkingsplots |
| `pyarrow` | Parquet-opslag |
| `emoji` | Emoji-detectie en categorisatie |
| `colorlog` | Gekleurde logging |

---

## Versie

Projectversie: 0.1.0
