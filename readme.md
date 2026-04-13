# DAV Chat Analysis

Naam: Lucas Joshua  
Studentnummer: 1905781

Python-project voor het vak Data Analysis & Visualisation. De repository bevat een pipeline die een WhatsApp-export inleest, opschoont, anonimiseert, features afleidt en visualisaties wegschrijft. Daarnaast is er een Streamlit-dashboard om de verwerkte data interactief te verkennen.

## Wat Dit Project Doet

- Leest een WhatsApp-chat uit `data/raw/raw_data.txt`
- Parseert berichten, afzenders en timestamps
- Anonimiseert afzenders en bewaart de mapping in `data/processed/user_mapping.csv`
- Verrijkt de data met emoji-, tijd- en incidentfeatures
- Slaat de verwerkte dataset op als CSV en Parquet
- Genereert de geselecteerde visualisaties in `img/`
- Biedt een dashboard via Streamlit

De parser is bewust generiek opgezet voor meerdere veelvoorkomende WhatsApp-exportformaten. Exports van Android, iPhone, Mac en Windows gebruiken niet altijd exact dezelfde datum-/tijdnotatie of voorlooptekens; de preprocessing probeert die varianten automatisch te herkennen.

## Snel Starten

Installeer dependencies:

```bash
uv sync
```

Draai de volledige pipeline:

```bash
uv run python -m src.main
```

Of geef expliciet een ander exportbestand mee:

```bash
uv run python -m src.main path/naar/chat_export.txt
```

Start het dashboard:

```bash
uv run streamlit run src/dashboard.py
```

## Input En Output

Verwachte input:

- `data/raw/raw_data.txt`
- of elk ander `.txt` exportbestand dat je als argument meegeeft aan `src.main`

Belangrijkste output:

- `data/processed/clean_chat_processed.csv`
- `data/processed/clean_chat_processed.parquet`
- `data/processed/user_mapping.csv`
- `img/les*/...` voor de gegenereerde figuren
- `logs/` voor logbestanden

## Projectstructuur

```text
dav-chat-analysis/
├── data/
│   ├── raw/
│   │   ├── raw_data.txt
│   │   └── names/
│   └── processed/
├── img/
├── logs/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── pipeline.py
│   ├── feature_pipeline.py
│   ├── dashboard.py
│   ├── modules/
│   │   ├── data_loader.py
│   │   ├── data_cleaning.py
│   │   ├── anonymizer.py
│   │   ├── preprocessor.py
│   │   ├── feature_engineering.py
│   │   └── author_stylometry.py
│   └── visualizations/
│       ├── registry.py
│       ├── utils.py
│       ├── plot_settings.py
│       ├── chat_activity.py
│       ├── emoji.py
│       ├── incident_timeline.py
│       ├── incident_context_modeling.py
│       ├── poisson_modeling.py
│       └── time_series_modeling.py
├── pyproject.toml
└── readme.md
```

## Waar Wat Staat

`src/main.py`

- klein entry point
- zet logging op
- start de pipeline op de standaard raw input

`src/pipeline.py`

- hoofdorkestratie van de pipeline
- bepaalt welke visualisaties standaard aan staan
- bepaalt welke features daarvoor nodig zijn
- slaat outputbestanden op

`src/feature_pipeline.py`

- houdt feature dependencies en execution order centraal
- voorkomt duplicatie in `pipeline.py`
- blijft expres klein: alleen resolve + apply

`src/modules/`

- preprocessing en feature engineering
- deze map bevat de pure datastappen van raw tekst naar bruikbare dataframe

`src/visualizations/`

- wrappers en low-level plotters
- `registry.py` laadt alleen de actieve pipeline-visualisaties
- `plot_settings.py` houdt stijlinstellingen consistent

`src/dashboard.py`

- interactieve Streamlit-app op basis van de verwerkte CSV
- gebruikt dezelfde processed data als de pipeline-output

## Technische Keuzes

### Waarom Een PipelineRunner?

De pipeline heeft meerdere opeenvolgende stappen met duidelijke volgorde: load, clean, anonymize, feature engineering, save, visualize. Een kleine orkestratieklasse maakt die flow expliciet zonder de codebase zwaar te maken.

### Waarom Een FeaturePipeline?

Feature engineering heeft dependencies, bijvoorbeeld `emoji_category` hangt af van `emoji_features`. Die dependency-logica stond eerst verspreid in `pipeline.py`. `FeaturePipeline` centraliseert dat op een simpele manier en houdt de hoofd-pipeline leesbaar.

### Waarom Functies In `modules/` In Plaats Van Overal Classes?

De meeste datastappen zijn pure transformaties op een `DataFrame`. Daar zijn losse functies vaak duidelijker dan extra classes. Classes zijn alleen gebruikt waar coördinatie of state echt helpt, zoals bij `PipelineRunner` en `ChatPreprocessor`.

### Waarom Zowel Matplotlib Als Plotly?

- `Matplotlib` is gebruikt voor statische PNG-output waar precieze controle en snelle export belangrijk zijn.
- `Plotly` is gebruikt waar interactieve varianten of complexere layout-handling handig zijn.
- `Kaleido` verzorgt de PNG-export van Plotly-figuren.

### Waarom CSV En Parquet?

- CSV is makkelijk te inspecteren en bruikbaar in het dashboard
- Parquet is compacter en handiger voor verdere analyse

### Waarom Regex / Bag-of-Words Voor Incidentdetectie?

Voor deze opdracht is een transparante en uitlegbare aanpak belangrijker dan een zwaarder model. De incidentdetectie is daarom bewust simpel en reproduceerbaar gehouden.

### Hoe Generiek Is De WhatsApp-parser?

De loader en parser proberen bewust niet van één specifieke export uit te gaan:

- meerdere encodings worden geprobeerd (`utf-8`, `utf-8-sig`, `utf-16`, `cp1252`, `latin-1`)
- bracketed en unbracketed WhatsApp prefixes worden herkend
- slash- en hyphen-datums worden ondersteund
- 24-uurs en 12-uurs tijden worden ondersteund
- seconden zijn optioneel
- multiline berichten blijven samengevoegd tot één bericht

Daardoor is de pipeline minder gebonden aan één specifieke chat of één exportplatform.

## Visualisaties Configureren

De standaardselectie staat bovenin [src/pipeline.py](/Users/lucasjoshua/Documents/Opleiding/4_Data_Analysis_and_Visualisation/dav-chat-analysis/src/pipeline.py). Daar kun je:

- visualisaties aan of uit zetten via `DEFAULT_VISUALIZATION_SELECTIONS`
- de featurebehoefte per visualisatie aanpassen via `VISUALIZATION_FEATURES`

## Dashboard

Het dashboard leest `data/processed/clean_chat_processed.csv` en biedt:

- KPI-kaarten
- datumfilters
- interactieve Plotly-visualisaties
- een UI om geselecteerde pipeline-visualisaties opnieuw te genereren

## Opmerkingen

- `img/` is bedoeld als gegenereerde outputmap en niet als broncode
- de README is bewust technisch gehouden; het inhoudelijke verhaal van de opdracht hoort in het verslag
