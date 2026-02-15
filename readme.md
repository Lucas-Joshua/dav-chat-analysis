# DAV Chat Analysis

## Studentinformatie

Naam: Lucas Joshua\
Studentnummer: \[VUL HIER JE STUDENTNUMMER IN\]

## Projectbeschrijving

Dit project is ontwikkeld voor het vak Data Analysis & Visualisation
(DAV).

De applicatie analyseert een WhatsApp chat-export (.txt) en voert de
volgende stappen uit:

-   Inlezen van ruwe chatdata
-   Opschonen en structureren van berichten
-   Anonimiseren van gebruikersnamen
-   Feature engineering (zoals detectie van links en scheldwoorden)
-   Genereren van metadata
-   Opslaan van resultaten in CSV en Parquet
-   Visualisatie van chatstatistieken

De pipeline is modulair opgebouwd en reproduceerbaar.

------------------------------------------------------------------------

## Projectstructuur

    dav-chat-analysis/
    ├── src/                # Broncode
    ├── data/
    │   ├── raw/            # Onbewerkte data
    │   └── processed/      # Verwerkte data (output)
    ├── img/                # Gegenereerde visualisaties
    ├── logs/               # Logbestanden
    ├── notebooks/          # Jupyter notebooks
    ├── pyproject.toml
    └── README.md

------------------------------------------------------------------------

## Installatie

Gebruik `uv` om de omgeving op te zetten:

``` bash
uv sync
```

------------------------------------------------------------------------

## Uitvoeren van de pipeline

Start de applicatie via:

``` bash
uv run python -m src.main
```

Na uitvoering worden:

-   Verwerkte data opgeslagen in `data/processed/`
-   Visualisaties opgeslagen in `img/`
-   Logs opgeslagen in `logs/`

------------------------------------------------------------------------

## Gebruikte technologieën

-   Python 3.11
-   pandas
-   matplotlib
-   pyarrow
-   colorlog

------------------------------------------------------------------------

## Versie

Projectversie: 0.1.0
