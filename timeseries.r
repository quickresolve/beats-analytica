# =============================================================================
# Time-Series Analysis: How Genre Distribution & Audio Features Have Changed
#
# Uses track_album_release_date to track:
#   1. Genre share over time (proportional stack + per-genre trend lines)
#   2. Audio feature trajectories per genre (smoothed yearly means)
#   3. Genre diversity index over time (Shannon entropy)
#   4. Genre pairwise audio-feature distance over time (do genres converge?)
#
# Requirements:  install.packages(c("ggplot2","dplyr","tidyr","scales","vegan"))
# =============================================================================

# ── 0. Packages ───────────────────────────────────────────────────────────────
for (pkg in c("ggplot2", "dplyr", "tidyr", "scales", "vegan")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# ── 1. Load & prepare data ────────────────────────────────────────────────────
FEATURES <- c("energy", "tempo", "danceability", "loudness", "liveness",
              "valence", "speechiness", "instrumentalness", "acousticness",
              "key", "mode", "time_signature", "duration_ms")

high <- read.csv("spotify-music-dataset/high_popularity_spotify_data.csv",
                 stringsAsFactors = FALSE)
low  <- read.csv("spotify-music-dataset/low_popularity_spotify_data.csv",
                 stringsAsFactors = FALSE)

data <- rbind(high[, c(FEATURES, "playlist_genre", "track_album_release_date",
                        "track_popularity")],
              low[,  c(FEATURES, "playlist_genre", "track_album_release_date",
                        "track_popularity")])
data <- na.omit(data)
data$playlist_genre       <- trimws(tolower(data$playlist_genre))
data$track_album_release_date <- trimws(data$track_album_release_date)

# Parse release year robustly (handles YYYY, YYYY-MM, YYYY-MM-DD)
data$year <- as.integer(substr(data$track_album_release_date, 1, 4))
data <- data[!is.na(data$year) & data$year >= 1954 & data$year <= 2024, ]

# ── 2. Keep genres with enough data; focus on 1990 onwards ───────────────────
MIN_TRACKS_GENRE <- 30       # min total tracks to include a genre
YEAR_START       <- 1990     # earlier years are too sparse to be meaningful

genre_totals <- sort(table(data$playlist_genre), decreasing = TRUE)
keep_genres  <- names(genre_totals[genre_totals >= MIN_TRACKS_GENRE])

ts_data <- data[data$playlist_genre %in% keep_genres &
                  data$year >= YEAR_START, ]
ts_data$playlist_genre <- factor(ts_data$playlist_genre)

cat(sprintf("Working with %d genres, %d tracks (%d–%d)\n",
            length(keep_genres), nrow(ts_data), YEAR_START, 2024))
cat("Genres:", paste(sort(keep_genres), collapse = ", "), "\n\n")

# ── 3. Yearly aggregates ──────────────────────────────────────────────────────

# 3a. Track counts per genre per year
yearly_counts <- ts_data |>
  count(year, playlist_genre, name = "n") |>
  group_by(year) |>
  mutate(total     = sum(n),
         share     = n / total) |>
  ungroup()

# 3b. Mean audio features per genre per year (only years with >= 5 tracks)
yearly_features <- ts_data |>
  group_by(year, playlist_genre) |>
  filter(n() >= 5) |>
  summarise(across(all_of(FEATURES), mean, .names = "{.col}"),
            n = n(),
            .groups = "drop")

# 3c. Overall yearly feature means (all genres combined)
yearly_overall <- ts_data |>
  group_by(year) |>
  filter(n() >= 10) |>
  summarise(across(all_of(FEATURES), mean, .names = "{.col}"),
            n_tracks = n(),
            .groups = "drop")

# 3d. Shannon diversity of genre mix per year
yearly_diversity <- yearly_counts |>
  group_by(year) |>
  summarise(
    n_genres  = n(),
    shannon   = -sum(share * log(share + 1e-12)),
    total     = first(total),
    .groups   = "drop"
  )

# ── 4. Plots ──────────────────────────────────────────────────────────────────
theme_ts <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(colour = "grey40", size = 10),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

# Colour palette (up to 20 genres)
n_gen  <- length(keep_genres)
pal    <- if (n_gen <= 8)  scales::hue_pal()(n_gen) else
          if (n_gen <= 12) RColorBrewer::brewer.pal(min(n_gen, 12), "Paired") else
          scales::hue_pal(c = 70, l = 55)(n_gen)

# ── Plot 1: Proportional area chart (genre share over time) ──────────────────
p1 <- ggplot(yearly_counts,
             aes(x = year, y = share, fill = playlist_genre)) +
  geom_area(position = "stack", colour = "white", linewidth = 0.15) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = c(0, 0)) +
  scale_x_continuous(breaks = seq(1990, 2025, 5), expand = c(0, 0)) +
  scale_fill_manual(values = pal, name = "Genre") +
  labs(title    = "Genre Share Over Time",
       subtitle = sprintf("Proportional composition of %d genres (%d–2024)",
                          length(keep_genres), YEAR_START),
       x = "Release Year", y = "Share of Tracks") +
  theme_ts

# ── Plot 2: Individual genre trend lines (absolute track count) ───────────────
p2 <- ggplot(yearly_counts,
             aes(x = year, y = n, colour = playlist_genre)) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  geom_point(size = 1.2, alpha = 0.6) +
  scale_colour_manual(values = pal, name = "Genre") +
  scale_x_continuous(breaks = seq(1990, 2025, 5)) +
  labs(title    = "Track Count per Genre Over Time",
       subtitle = "Raw counts — note the surge in catalogue from 2018 onwards",
       x = "Release Year", y = "Number of Tracks") +
  theme_ts

# ── Plot 3: Shannon diversity index over time ─────────────────────────────────
p3 <- ggplot(yearly_diversity, aes(x = year, y = shannon)) +
  geom_col(aes(fill = total), width = 0.8) +
  geom_smooth(method = "loess", span = 0.4, se = TRUE,
              colour = "#e74c3c", linewidth = 1) +
  scale_fill_gradient(low = "#d5e8f7", high = "#2980b9",
                      name = "Total\ntracks", labels = comma) +
  scale_x_continuous(breaks = seq(1990, 2025, 5)) +
  labs(title    = "Genre Diversity Over Time (Shannon Entropy)",
       subtitle = "Higher = more evenly distributed across genres",
       x = "Release Year", y = "Shannon H′") +
  theme_ts

# ── Plot 4: Key audio features over time (all-genre aggregate) ────────────────
feature_long <- yearly_overall |>
  pivot_longer(cols = c(energy, danceability, valence, acousticness,
                        speechiness, instrumentalness),
               names_to = "feature", values_to = "mean_value")

p4 <- ggplot(feature_long, aes(x = year, y = mean_value, colour = feature)) +
  geom_line(linewidth = 0.7, alpha = 0.5) +
  geom_smooth(method = "loess", span = 0.5, se = FALSE, linewidth = 1.2) +
  facet_wrap(~ feature, scales = "free_y", ncol = 2) +
  scale_x_continuous(breaks = seq(1990, 2025, 10)) +
  scale_colour_brewer(palette = "Set2", guide = "none") +
  labs(title    = "Mean Audio Features Over Time (All Genres)",
       subtitle = "LOESS trend; shaded = 95% CI",
       x = "Release Year", y = "Mean Value") +
  theme_ts

# ── Plot 5: Energy & Valence per genre over time (faceted) ────────────────────
# Use only top 6 genres by total count for readability
top6 <- names(sort(table(ts_data$playlist_genre), decreasing = TRUE))[1:min(6, n_gen)]

p5_data <- yearly_features |>
  filter(playlist_genre %in% top6) |>
  select(year, playlist_genre, energy, valence, danceability) |>
  pivot_longer(cols = c(energy, valence, danceability),
               names_to = "feature", values_to = "value")

p5 <- ggplot(p5_data, aes(x = year, y = value, colour = feature)) +
  geom_line(alpha = 0.4, linewidth = 0.6) +
  geom_smooth(method = "loess", span = 0.6, se = FALSE, linewidth = 1.1) +
  facet_wrap(~ playlist_genre, ncol = 3) +
  scale_colour_manual(values = c(energy = "#e74c3c",
                                 valence = "#2ecc71",
                                 danceability = "#3498db"),
                      name = "Feature") +
  scale_x_continuous(breaks = seq(1990, 2025, 10)) +
  labs(title    = "Energy, Valence & Danceability per Genre Over Time",
       subtitle = sprintf("Top %d genres by track count; LOESS trend", length(top6)),
       x = "Release Year", y = "Mean Value (0–1)") +
  theme_ts

# ── Plot 6: Genre convergence — pairwise Euclidean distance in feature space ─
# For each year (>= 10 tracks per genre), compute centroid distances

norm_cols <- c("energy", "danceability", "valence",
               "acousticness", "speechiness", "loudness")

# Normalise features to [0,1] globally for fair distance computation
norm_df <- ts_data
for (col in norm_cols) {
  rng <- range(norm_df[[col]], na.rm = TRUE)
  if (diff(rng) > 0)
    norm_df[[col]] <- (norm_df[[col]] - rng[1]) / diff(rng)
}

centroids <- norm_df |>
  group_by(year, playlist_genre) |>
  filter(n() >= 10) |>
  summarise(across(all_of(norm_cols), mean, .names = "{.col}"),
            .groups = "drop")

# Mean pairwise distance between genre centroids per year
divergence_by_year <- centroids |>
  group_by(year) |>
  filter(n() >= 3) |>       # at least 3 genres to compute meaningful distances
  summarise({
    mat  <- as.matrix(.[, norm_cols])
    dmat <- dist(mat)
    tibble(mean_dist = mean(dmat),
           sd_dist   = sd(dmat),
           n_genres  = nrow(mat))
  }, .groups = "drop")

p6 <- ggplot(divergence_by_year, aes(x = year, y = mean_dist)) +
  geom_ribbon(aes(ymin = mean_dist - sd_dist,
                  ymax = mean_dist + sd_dist),
              fill = "#9b59b6", alpha = 0.2) +
  geom_line(colour = "#9b59b6", linewidth = 1) +
  geom_point(aes(size = n_genres), colour = "#9b59b6", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, colour = "#2c3e50",
              linetype = "dashed", linewidth = 0.8) +
  scale_size_continuous(range = c(1.5, 4), name = "# Genres") +
  scale_x_continuous(breaks = seq(1990, 2025, 5)) +
  labs(title    = "Are Genres Converging in Audio Feature Space?",
       subtitle = "Mean pairwise Euclidean distance between genre centroids\n(ribbon = ±1 SD; dashed = linear trend; point size = genres present)",
       x = "Release Year", y = "Mean Inter-Genre Distance") +
  theme_ts

# ── 5. Print all plots ────────────────────────────────────────────────────────
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)

# ── 6. Numeric summary ────────────────────────────────────────────────────────
cat("\n── Genre share at key epochs ──────────────────────────────────────────\n")
epochs <- c(1995, 2000, 2005, 2010, 2015, 2020, 2024)
for (yr in epochs) {
  row <- yearly_counts |>
    filter(year == yr) |>
    arrange(desc(share)) |>
    slice_head(n = 3)
  if (nrow(row) == 0) {
    cat(sprintf("%d: no data\n", yr)); next
  }
  top_str <- paste(sprintf("%s %.0f%%", row$playlist_genre, row$share * 100),
                   collapse = "  │  ")
  cat(sprintf("%d  top 3: %s\n", yr, top_str))
}

cat("\n── Overall audio feature trends (1990–2024, linear slope × 10 years) ──\n")
years_range <- range(yearly_overall$year)
for (feat in c("energy", "danceability", "valence", "acousticness",
               "speechiness", "instrumentalness")) {
  sub <- yearly_overall[, c("year", feat)]
  sub <- sub[complete.cases(sub), ]
  if (nrow(sub) < 5) next
  m   <- lm(reformulate("year", feat), data = sub)
  slp <- coef(m)[["year"]] * 10
  cat(sprintf("  %-18s slope per decade: %+.4f\n", feat, slp))
}

cat("\n── Diversity trend ────────────────────────────────────────────────────\n")
div_sub <- yearly_diversity[yearly_diversity$total >= 10, ]
div_lm  <- lm(shannon ~ year, data = div_sub)
cat(sprintf("  Shannon H′ slope per year: %+.4f  (p = %.3f)\n",
            coef(div_lm)[["year"]],
            summary(div_lm)$coefficients["year", "Pr(>|t|)"]))

cat("\n── Genre convergence trend ─────────────────────────────────────────────\n")
conv_lm <- lm(mean_dist ~ year, data = divergence_by_year)
cat(sprintf("  Inter-genre distance slope per year: %+.5f  (p = %.3f)\n",
            coef(conv_lm)[["year"]],
            summary(conv_lm)$coefficients["year", "Pr(>|t|)"]))
cat(sprintf("  Direction: %s\n",
            ifelse(coef(conv_lm)[["year"]] < 0,
                   "CONVERGING (genres sounding more similar over time)",
                   "DIVERGING  (genres sounding more distinct over time)")))
