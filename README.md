# Beats Analytica — Analysis Documentation

Two R scripts analyse the Spotify music dataset from `./spotify-music-dataset/`:
`fewshot.r` benchmarks LLM-based genre classification against a Random Forest,
and `timeseries.r` tracks how genre composition and audio features have shifted
over seven decades of recorded music.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Dataset Overview](#dataset-overview)
3. [Setup & Dependencies](#setup--dependencies)
4. [fewshot.r — Few-Shot Genre Classification](#fewhoshotr--few-shot-genre-classification)
5. [timeseries.r — Genre & Feature Trends Over Time](#timeseriesr--genre--feature-trends-over-time)
6. [Design Decisions](#design-decisions)

---

## Repository Layout

```
beats-analytica/
├── spotify-music-dataset/
│   ├── high_popularity_spotify_data.csv
│   └── low_popularity_spotify_data.csv
├── fewshot.r          # LLM vs Random Forest classification
├── timeseries.r       # Temporal trend analysis
└── ANALYSIS.md        # This file
```

---

## Dataset Overview

| File | Rows | Description |
|------|------|-------------|
| `high_popularity_spotify_data.csv` | 1 686 | Tracks with high Spotify popularity scores |
| `low_popularity_spotify_data.csv` | 3 145 | Tracks with low Spotify popularity scores |

Both files share the same 29 columns, though column order differs between them.
The 13 numeric audio features used across both scripts are:

| Feature | Range | Description |
|---------|-------|-------------|
| `energy` | 0–1 | Perceptual intensity and activity |
| `tempo` | BPM | Estimated beats per minute |
| `danceability` | 0–1 | Suitability for dancing |
| `loudness` | dB (neg.) | Overall track loudness |
| `liveness` | 0–1 | Probability of a live performance |
| `valence` | 0–1 | Musical positivity / happiness |
| `speechiness` | 0–1 | Presence of spoken words |
| `instrumentalness` | 0–1 | Likelihood of no vocals |
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `key` | 0–11 | Estimated musical key (pitch class) |
| `mode` | 0 or 1 | Major (1) or minor (0) |
| `time_signature` | integer | Estimated beats per measure |
| `duration_ms` | ms | Track duration |

The target variable is `playlist_genre`. After combining both files and
filtering to genres with ≥ 30 samples, there are roughly 15–20 distinct genres
spanning pop, rock, hip-hop, electronic, latin, r&b, ambient, metal, folk, and
more.

---

## Setup & Dependencies

### R version

R ≥ 4.1 is recommended.

### Required packages

Both scripts auto-install missing packages on first run. To install manually:

```r
install.packages(c(
  # fewshot.r
  "httr2", "jsonlite", "randomForest",
  # timeseries.r
  "ggplot2", "dplyr", "tidyr", "scales", "vegan"
))
```

### API key (fewshot.r only)

`fewshot.r` calls the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages).
Set your key before sourcing the script:

```r
Sys.setenv(ANTHROPIC_API_KEY = "sk-ant-...")
```

Or add it to your `.Renviron` file so it persists across sessions:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Running the scripts

Set your working directory to the repo root, then source either file:

```r
setwd("/path/to/beats-analytica")

# Classification benchmark
Sys.setenv(ANTHROPIC_API_KEY = "sk-ant-...")
source("fewshot.r")

# Time-series analysis (no API key needed)
source("timeseries.r")
```

Both scripts print progress and a numeric summary to the console. `timeseries.r`
also renders six `ggplot2` plots to the active graphics device.

---

## fewshot.r — Few-Shot Genre Classification

### Goal

Evaluate whether a large language model can classify music genres from numeric
audio features alone, using only a handful of labelled examples (few-shot
prompting), and compare that accuracy to a trained Random Forest.

### Pipeline

```
Raw CSVs
  └─ combine + normalise column names
       └─ filter to genres ≥ 30 samples
            └─ stratified 80/20 train/test split
                 ├─ Random Forest  (trained on full training set)
                 └─ Few-shot LLM   (2 examples/genre → API prompt)
                      └─ compare accuracy, per-genre breakdown, confusion matrix
```

### Analyses

#### 1. Random Forest baseline

A 300-tree Random Forest is trained on all 13 numeric features using the full
training split. Predictions on the test set establish the baseline accuracy that
the LLM must beat.

Variable importance (Mean Decrease Gini) is printed at the end to show which
audio features are most discriminative for genre classification.

#### 2. LLM few-shot classification

Two examples per genre are sampled from the training set and formatted as
compact feature strings:

```
[energy=0.83 tempo=128 dance=0.71 loud=-5.2dB ... dur=214s] → rock
```

These examples, plus the list of valid genre names, are prepended to every
classification prompt. Test tracks are sent to the API in **batches of 5** to
reduce cost. The model (`claude-opus-4-6`) is asked to return a numbered list
of genre names — one per track — and nothing else.

Response parsing strips leading numbering and whitespace, lowercases the output,
and attempts an exact match against the known genre list. A substring fallback
handles minor variations (e.g. `"hip hop"` → `"hip-hop"`). Lines that cannot
be matched are recorded as `NA` and excluded from accuracy calculations.

#### 3. Results reported

| Output | Description |
|--------|-------------|
| Overall accuracy | LLM vs RF on the same test subsample |
| Lift | Percentage-point difference (LLM − RF) |
| Per-genre table | Accuracy for each genre, both methods side by side |
| LLM confusion matrix | Full predicted × actual table |
| RF feature importances | Top 7 features ranked by Gini decrease |

### Cost estimate

With `MAX_TEST_N = 100` and `BATCH_SIZE = 5` (defaults), the script sends
~20 API calls to `claude-opus-4-6`. Each prompt is approximately 7 000 input
tokens (few-shot examples × genres) and requests at most 256 output tokens.
Estimated cost: **< $1 USD** per full run at May 2026 pricing.

Raise `MAX_TEST_N` to `NULL` to evaluate the full test set.

---

## timeseries.r — Genre & Feature Trends Over Time

### Goal

Use `track_album_release_date` to answer: how has the genre composition of
popular music changed since 1990, and are the audio characteristics of genres
becoming more similar or more distinct over time?

### Pipeline

```
Raw CSVs
  └─ parse release year from mixed date formats (YYYY / YYYY-MM / YYYY-MM-DD)
       └─ filter to 1990–2024, genres ≥ 30 samples
            └─ yearly aggregates (counts, means, diversity, centroids)
                 └─ six plots + numeric summaries
```

### Analyses

#### Plot 1 — Genre share over time (proportional area chart)

Each year's tracks are decomposed into genre proportions, stacked to 100%.
This separates genuine shifts in genre popularity from the overall growth of
the catalogue (which is dominated by 2020–2024 releases).

#### Plot 2 — Track count per genre over time (line chart)

Absolute counts make the sharp catalogue expansion after 2018 visible. A genre
whose share is flat but whose count is rising is growing with the market;
a genre whose count is flat but whose share is falling is being crowded out.

#### Plot 3 — Genre diversity index over time (Shannon entropy)

Shannon entropy H′ = −Σ pᵢ ln(pᵢ) is computed across genre proportions each
year. A rising H′ means the genre mix is becoming more even; a falling H′
means one or two genres are dominating. A LOESS trend line and a linear
regression slope (with p-value) are printed to the console.

#### Plot 4 — Six audio features over time, all genres combined

Yearly means of `energy`, `danceability`, `valence`, `acousticness`,
`speechiness`, and `instrumentalness` are plotted with individual LOESS
smoothers. This captures the well-documented trend of popular music becoming
louder and more energetic over time, and whether that trend continues into
the mid-2020s.

Each feature gets its own free-scaled facet so slow-moving features like
`instrumentalness` are not dwarfed by faster-moving ones.

#### Plot 5 — Energy, valence & danceability per genre (faceted lines)

The same three features are broken out for the **top 6 genres by track count**,
showing whether feature trends are genre-specific or universal. For example,
hip-hop may become more melodic (higher valence) while metal stays low — or
they may converge.

Only genre-years with ≥ 5 tracks are included to suppress noise.

#### Plot 6 — Inter-genre convergence in audio feature space

For each year, genre centroids are computed in a six-dimensional normalised
feature space. The mean pairwise Euclidean distance between all genre centroids
is then plotted over time, with ±1 SD ribbon and a linear trend line.

A **declining distance** means genres are sounding more alike — convergence.
A **rising distance** means genres are becoming more acoustically distinct —
divergence. The slope and p-value are printed to the console with a plain-
language direction label.

Features are normalised globally to [0, 1] before distance computation so that
`loudness` (in dB, spanning roughly −30 to 0) does not dominate the metric.

### Numeric summaries printed to console

| Summary | What it shows |
|---------|---------------|
| Top-3 genres at key epochs | Dominant genres in 1995, 2000, 2005, 2010, 2015, 2020, 2024 |
| Feature slopes per decade | How fast each audio feature is changing on average |
| Shannon diversity slope | Linear trend in genre evenness, with p-value |
| Convergence direction | Whether genres are sonically converging or diverging |

---

## Design Decisions

### Both scripts

**Minimum genre sample size (30 tracks)**
Genres with fewer than 30 tracks across the combined dataset are dropped.
Small-sample genres produce unreliable feature averages and inflate accuracy
estimates in both classifiers.

**Column selection by name, not position**
The two CSV files have the same columns in a different order. All subsetting
uses explicit column names (e.g. `high[, c(FEATURES, TARGET)]`) rather than
integer indices to avoid silent misalignment.

**`na.omit` applied after column selection**
Dropping NA rows before selecting columns would remove rows that are only
missing in columns we never use (e.g. `analysis_url`). Selecting first keeps
the usable data.

---

### fewshot.r

**Few-shot count (N_SHOT = 2)**
Two examples per genre keeps each prompt under ~7 000 tokens, which provides
meaningful headroom below the per-minute token rate limit. The original value
of 3 was reduced after hitting the 30 000 token/minute org limit in practice;
2 examples still gives the model a clear signal per genre while reducing
prompt size by ~30%. Increasing to 5–10 examples would likely improve LLM
accuracy but further tightens the rate-limit budget.

**Batch size (BATCH_SIZE = 5)**
Batching multiple tracks into one API call amortises the few-shot prefix cost.
With 5 tracks per call, approximately 95% of each prompt is the fixed few-shot
block — reducing effective per-track cost by ~5× compared to one call per track.

**Capped test set (MAX_TEST_N = 100)**
The default caps LLM evaluation at 100 stratified test samples to bound API
spend. The Random Forest is still evaluated on the full test set for the
headline accuracy figure, and on the same 100-sample subsample for the direct
comparison.

**Fuzzy response parsing**
The LLM occasionally adds punctuation, articles, or slight capitalization
differences. The parser strips leading `N.` / `N)` prefixes, lowercases, and
falls back to substring matching before recording `NA`. Genre names with
hyphens (e.g. `hip-hop`) are the most common mismatch.

**Model choice (`claude-opus-4-6`)**
Opus 4.6 is used to maximise classification quality. Switching to
`claude-haiku-4-5` would cut API cost by ~5× at the expense of likely lower
accuracy — a useful ablation to run if cost is a constraint.

**Rate limiting (`throttle` + `call_claude` retry)**
The Anthropic API enforces a 30 000 input-token-per-minute limit at the org
level. Running 20 batches with a 0.4-second fixed delay was enough to exhaust
this budget in under 10 seconds. Two complementary mechanisms now prevent this:

*Rolling-window token budget (`throttle`)*
Before every API call, `throttle` estimates the prompt size (~3.5 characters
per token) and sums the tokens logged in the previous 60 seconds. If adding
the new prompt would exceed `TOKEN_BUDGET` (20 000 — a 10 000-token margin
below the hard limit), it calculates which logged entries need to expire to
create enough headroom, then sleeps until that point plus a 1-second buffer.
Critically, this runs in a `repeat` loop: after each sleep it re-checks from
scratch, so a single expiry that still leaves insufficient headroom causes
another targeted sleep rather than firing the request early. The loop only
exits once the budget check genuinely passes.

The 20 000-token budget (rather than, say, 28 000) exists because the
`nchar / 3.5` heuristic can undercount tokens containing multi-byte characters
or unusual formatting. The extra margin absorbs that estimation error.

*429 retry with server-specified backoff (`call_claude`)*
If a 429 slips through despite the proactive limiter (e.g. another process on
the same org key is consuming tokens), `call_claude` reads the `retry-after`
response header for the server-mandated wait, adds 5 seconds, and retries up
to 4 times before raising an error. Previously, any 429 immediately produced
`NA` for the entire batch; now it recovers silently. The console prints a
message for each retry so progress is still visible.

---

### timeseries.r

**Year range (1990–2024)**
Years before 1990 have fewer than 25 tracks in the combined dataset, making
per-genre yearly averages statistically unreliable. The cut-off is hard-coded
in `YEAR_START` and can be lowered if the dataset is enriched with older
recordings.

**Minimum tracks per genre-year for feature plots (5)**
Plot 5 and the convergence analysis filter out genre-year cells with fewer
than 5 (or 10) observations to prevent a single outlier track from dominating
the trend line for that year.

**Global normalisation for distance computation**
`loudness` is measured in negative dB (typically −30 to 0) while most other
features are bounded 0–1. Without normalisation, loudness would account for
the majority of every pairwise distance. Each feature is scaled to [0, 1]
using the global min/max across all tracks and years before centroid distances
are computed.

**Shannon entropy rather than simple genre count**
Counting distinct genres per year would reward years with many rare genres
equally to years with genuinely even genre representation. Shannon entropy
weights by proportion, so a year where 10 genres each hold 10% scores higher
than a year where one genre holds 90% and nine others share the rest.

**LOESS smoothing (span = 0.4–0.6)**
Raw yearly means are noisy, especially before 2010 when track counts per year
are low. LOESS with moderate span smooths over single-year outliers while
preserving decade-scale inflections. The span values are noted in the code and
can be widened for smoother curves or narrowed to emphasise year-to-year
variation.
